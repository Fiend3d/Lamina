#pragma once
#include "lamina/model/cuda_projection.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace lamina::model {
// Persistent pool; an entire expert is one task so intermediate activations
// remain worker-local. Futures propagate errors before any result is combined.
class CpuExperts {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> jobs_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
public:
    explicit CpuExperts(unsigned workers);
    ~CpuExperts();
    std::future<std::vector<float>> submit(MoeWeights weights, int expert,
                                          std::shared_ptr<const std::vector<float>> input);
};
}
