#pragma once
#include "lamina/model/cuda_projection.hpp"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lamina::model {
// Persistent pool that runs one token's CPU experts. Every expert is split by
// output rows across all workers: gate/up chunks first, then, once an expert's
// last gate/up chunk finishes, its SwiGLU and down chunks. Each output row is
// computed exactly as a whole-expert task would compute it, so results do not
// depend on the worker count. Workers spin briefly between batches because a
// decode layer issues a batch about every millisecond.
class CpuExperts {
public:
    explicit CpuExperts(unsigned workers, bool fast = false);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    // Starts the routed experts `ids` for one input row of the expert width.
    // outputs[i] receives the down projection of ids[i]. `input` and every
    // output must stay valid until wait() returns. One batch at a time.
    void start(const MoeWeights& weights, const std::vector<int>& ids, const float* input,
               const std::vector<float*>& outputs);
    // Helps run the current batch on the calling thread and returns once it is
    // complete, rethrowing the first worker error.
    void wait();

private:
    struct Item { int expert; int matrix; int begin; int end; };  // matrix: 0 gate, 1 up, 2 down
    struct ExpertState {
        std::vector<float> gate, up, swiglu;
        std::vector<uint8_t> swiglu_q;
        std::atomic<int> pending{0};
        std::atomic<bool> ready{false};
    };
    static constexpr size_t kIdle = ~size_t(0) >> 1;

    void worker_loop();
    void work();
    void run(const Item& item);
    void finish_gate_up(int expert);

    bool fast_ = false;
    MoeWeights weights_;
    std::vector<int> ids_;
    const float* input_ = nullptr;
    std::vector<float*> outputs_;
    std::vector<uint8_t> input_q_, input_q_up_;
    std::vector<Item> items_;
    std::vector<std::unique_ptr<ExpertState>> experts_;
    std::atomic<size_t> next_{kIdle}, count_{0}, finished_{0};
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool> failed_{false}, stopping_{false};
    std::exception_ptr error_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<std::thread> workers_;
};
}
