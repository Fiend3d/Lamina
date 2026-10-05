#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Lamina CUDA runtime: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("Lamina CUDA devices: %d\n", devices);
    return 0;
}
