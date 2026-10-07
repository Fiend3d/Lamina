#pragma once

// Worker-backed pinned staging follows Strata's Stager design in
// src/prefill/prefill.cpp (MIT). Host copies run ahead of DMA; issued sequence
// numbers prevent a worker from reading a stale event when a ring slot wraps.
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <cstring>
#include <future>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>

namespace lamina::model {
class WeightStager {
public:
    static constexpr unsigned slots = 16;
    static constexpr size_t bytes = 4 * 1024 * 1024;
    struct Ticket { uint64_t sequence; std::future<void> ready; };
    std::array<uint8_t*, slots> host{};
    std::array<cudaEvent_t, slots> done{};
private:
    std::mutex mutex_;
    std::condition_variable wake_;
    std::queue<std::function<void()>> jobs_;
    std::vector<std::thread> workers_;
    uint64_t next_ = 0, issued_ = 0;
    std::set<uint64_t> submitted_;
    bool stopping_ = false;
    static void check(cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
public:
    WeightStager() {
        try {
            for (unsigned i = 0; i < slots; ++i) {
                check(cudaMallocHost(reinterpret_cast<void**>(&host[i]), bytes));
                check(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
            }
            for (unsigned i = 0; i < 2; ++i) workers_.emplace_back([this] {
                cudaSetDevice(0);
                for (;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock lock(mutex_);
                        wake_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
                        if (stopping_ && jobs_.empty()) return;
                        job = std::move(jobs_.front()); jobs_.pop();
                    }
                    job();
                }
            });
        } catch (...) { stop(); throw; }
    }
    ~WeightStager() { stop(); }
    void stop() {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        wake_.notify_all();
        for (auto& thread : workers_) if (thread.joinable()) thread.join();
        workers_.clear();
        for (unsigned i = 0; i < slots; ++i) {
            if (done[i]) cudaEventSynchronize(done[i]);
            if (done[i]) cudaEventDestroy(done[i]);
            if (host[i]) cudaFreeHost(host[i]);
            done[i] = nullptr; host[i] = nullptr;
        }
    }
    Ticket prepare(const uint8_t* source, size_t count) {
        if (!source || count > bytes) throw std::invalid_argument("invalid staging job");
        std::lock_guard lock(mutex_);
        const uint64_t sequence = next_++;
        auto task = std::make_shared<std::packaged_task<void()>>([this, source, count, sequence] {
            if (sequence >= slots) {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stopping_ || issued_ > sequence - slots; });
                if (stopping_) throw std::runtime_error("weight staging stopped");
                lock.unlock();
                check(cudaEventSynchronize(done[sequence % slots]));
            }
            std::memcpy(host[sequence % slots], source, count);
        });
        auto future = task->get_future();
        jobs_.emplace([task] { (*task)(); }); wake_.notify_all();
        return {sequence, std::move(future)};
    }
    void issued(uint64_t sequence, cudaStream_t stream) {
        check(cudaEventRecord(done[sequence % slots], stream));
        {
            std::lock_guard lock(mutex_);
            submitted_.insert(sequence);
            while (submitted_.erase(issued_)) ++issued_;
        }
        wake_.notify_all();
    }
};
}
