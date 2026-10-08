#pragma once
#include "lamina/model/cuda_projection.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lamina::model {
// Persistent pool that runs one layer's CPU experts for one or two tokens.
// Every expert is split by output rows across all workers: gate/up chunks
// first, then, once an expert's last gate/up chunk finishes, its SwiGLU and
// down chunks. A weight row is read once and applied to every token that
// routed the expert, so a two-token batch reads each shared expert from RAM
// once. Each output row is computed exactly as a whole-expert task would
// compute it, so results do not depend on the worker count or the token count.
// Workers spin briefly between batches because a decode layer issues a batch
// about every millisecond.
class CpuExperts {
public:
    static constexpr int kMaxTokens = 2;

    explicit CpuExperts(unsigned workers, bool fast = false);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    // Starts the routed experts `ids` for one input row of the expert width.
    // outputs[i] receives the down projection of ids[i]. `input` and every
    // output must stay valid until wait() returns. One batch at a time.
    void start(const MoeWeights& weights, const std::vector<int>& ids, const float* input,
               const std::vector<float*>& outputs);
    // Two-token form: outputs[i][t] receives expert ids[i] applied to token t's
    // input, or is null when token t did not route that expert.
    void start(const MoeWeights& weights, const std::vector<int>& ids,
               const std::array<const float*, kMaxTokens>& inputs, int tokens,
               const std::vector<std::array<float*, kMaxTokens>>& outputs);
    // Helps run the current batch on the calling thread and returns once it is
    // complete, rethrowing the first worker error.
    void wait();

    // Diagnostics: cumulative batch time from start() to the last finished
    // item, time until the first item was claimed, and batch/expert counts.
    struct Stats { double batch_ms = 0, first_claim_ms = 0; uint64_t batches = 0, experts = 0; };
    Stats stats() const { return stats_; }

private:
    struct Item { int expert; int matrix; int begin; int end; };  // matrix: 0 gate, 1 up, 2 down
    struct ExpertState {
        std::array<std::vector<float>, kMaxTokens> gate, up, swiglu;
        std::array<std::vector<uint8_t>, kMaxTokens> swiglu_q;
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
    int tokens_ = 1;
    std::array<const float*, kMaxTokens> inputs_{};
    std::vector<std::array<float*, kMaxTokens>> outputs_;
    std::array<std::vector<uint8_t>, kMaxTokens> input_q_, input_q_up_;
    std::vector<Item> items_;
    std::vector<std::unique_ptr<ExpertState>> experts_;
    std::atomic<size_t> next_{kIdle}, count_{0}, finished_{0};
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool> failed_{false}, stopping_{false};
    std::atomic<int64_t> started_ns_{0}, first_claim_ns_{0};
    Stats stats_;
    std::exception_ptr error_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<std::thread> workers_;
};
}
