// FP32 tiled attention against an independent host calculation, including a
// 128K cache with its only nonzero value at the oldest position.
#include "lamina/model/cuda_kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LAMINA_PREFILL_BLAS
#include <cublas_v2.h>
#endif

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
#ifdef LAMINA_PREFILL_BLAS
void matrix_check() {
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    cublasHandle_t blas;
    const auto bc=[](cublasStatus_t s) { if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("matrix test cuBLAS failure"); };
    bc(cublasCreate(&blas)); bc(cublasSetStream(blas,stream)); bc(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));
    constexpr int columns=9;
    for(int n : {2057,131072}) {
        std::vector<float> q(columns*8192), k(size_t(n)*512), v(k.size());
        const bool distant=n==131072;
        for(size_t i=0;i<q.size();++i) q[i]=distant ? 0 : .17f*std::sin(float(i)*.043f);
        for(size_t i=0;i<k.size();++i) { k[i]=distant ? 0 : .21f*std::cos(float(i)*.012f); v[i]=distant ? 0 : std::sin(float(i)*.017f); }
        if(distant) for(int j=0;j<512;++j) v[j]=float(n*2*(j%7-3));
        float *dq,*dk,*dv,*pq,*pk,*pv,*scores,*meta,*output,*acc,*reference,*result;
        const auto alloc=[](float** p,size_t elements){check(cudaMalloc(p,elements*4));};
        alloc(&dq,q.size());alloc(&dk,k.size());alloc(&dv,v.size());alloc(&pq,q.size()/2);
        alloc(&pk,2048*512);alloc(&pv,2048*512);alloc(&scores,16*columns*2048);
        alloc(&meta,16*columns*2);alloc(&output,16*columns*256);alloc(&acc,16*columns*258);
        alloc(&reference,16*columns*256);alloc(&result,16*columns*256);
        check(cudaMemcpyAsync(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dk,k.data(),k.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dv,v.data(),v.size()*4,cudaMemcpyHostToDevice,stream));
        for(int begin=0;begin<n;begin+=2048)
            lamina::model::cuda::attn_fused_columns(dq,dk+size_t(begin)*512,dv+size_t(begin)*512,std::min(2048,n-begin),begin,n-columns,columns,acc,begin==0,false,stream);
        lamina::model::cuda::attn_columns_finish(dq,acc,columns,reference,stream);
        lamina::model::cuda::attn_matrix_queries(dq,columns,pq,false,stream);
        const float scale=.0625f,one=1,zero=0;
        for(int begin=0;begin<n;begin+=2048) {
            const int count=std::min(2048,n-begin);
            lamina::model::cuda::attn_matrix_kv(dk+size_t(begin)*512,dv+size_t(begin)*512,count,false,pk,pv,false,stream);
            for(int kv=0;kv<2;++kv)
                bc(cublasGemmStridedBatchedEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,count,columns,256,&scale,
                    pk+kv*count*256,CUDA_R_32F,256,0,pq+kv*8*columns*256,CUDA_R_32F,256,columns*256,
                    &zero,scores+kv*8*count*columns,CUDA_R_32F,count,count*columns,8,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
            lamina::model::cuda::attn_matrix_softmax(scores,count,columns,begin,n-columns,meta,nullptr,false,stream);
            for(int kv=0;kv<2;++kv)
                bc(cublasGemmStridedBatchedEx(blas,CUBLAS_OP_N,CUBLAS_OP_N,256,columns,count,&one,
                    pv+kv*count*256,CUDA_R_32F,256,0,scores+kv*8*count*columns,CUDA_R_32F,count,count*columns,
                    &zero,output+kv*8*columns*256,CUDA_R_32F,256,columns*256,8,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
            lamina::model::cuda::attn_matrix_merge(output,meta,columns,acc,begin==0,stream);
        }
        lamina::model::cuda::attn_columns_finish(dq,acc,columns,result,stream);
        std::vector<float> want(16*columns*256),got(want.size());
        check(cudaMemcpyAsync(want.data(),reference,want.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(got.data(),result,got.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
        double error=0;for(size_t i=0;i<got.size();++i)error=std::max(error,std::abs(double(got[i])-want[i]));
        std::printf("matrix attention n=%d columns=%d max_abs_diff=%.9g\n",n,columns,error);
        if(error>=1e-5)throw std::runtime_error("matrix attention parity failed");
        for(float* p : {dq,dk,dv,pq,pk,pv,scores,meta,output,acc,reference,result})check(cudaFree(p));
    }
    bc(cublasDestroy(blas));check(cudaStreamDestroy(stream));
}
#endif
void benchmark() {
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    constexpr int columns = 128, heads = 16, dim = 256, stride = 512;
    for (int n : {4096, 16384, 131072}) {
        std::vector<float> q(size_t(columns)*8192), k(size_t(n)*stride), v(k.size());
        for (size_t i = 0; i < q.size(); ++i) q[i] = .17f * std::sin(float(i)*.043f);
        for (size_t i = 0; i < k.size(); ++i) { k[i] = .21f * std::cos(float(i)*.012f); v[i] = std::sin(float(i)*.017f); }
        float *dq, *dk, *dv, *partial, *accum, *out;
        check(cudaMalloc(&dq,q.size()*4)); check(cudaMalloc(&dk,k.size()*4)); check(cudaMalloc(&dv,v.size()*4));
        check(cudaMalloc(&partial,size_t(columns)*heads*16*258*4));
        check(cudaMalloc(&accum,size_t(columns)*heads*258*4)); check(cudaMalloc(&out,size_t(columns)*heads*dim*4));
        check(cudaMemcpyAsync(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dk,k.data(),k.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dv,v.data(),v.size()*4,cudaMemcpyHostToDevice,stream));
        std::vector<float> reference(size_t(columns)*4096), result(reference.size());
        for (bool fused : {false,true}) {
            auto launch = [&] {
                for (int begin = 0; begin < n; begin += 2048) {
                    const int count = std::min(2048,n-begin);
                    if (fused) lamina::model::cuda::attn_fused_columns(dq,dk+size_t(begin)*stride,dv+size_t(begin)*stride,
                        count,begin,n-columns,columns,accum,begin==0,false,stream);
                    else lamina::model::cuda::attn_columns_tile(dq,dk+size_t(begin)*stride,dv+size_t(begin)*stride,
                        count,begin,n-columns,columns,partial,accum,begin==0,stream);
                }
                lamina::model::cuda::attn_columns_finish(dq,accum,columns,out,stream);
            };
            launch(); check(cudaStreamSynchronize(stream));
            cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
            check(cudaEventRecord(start,stream));
            for (int repeat=0; repeat<3; ++repeat) launch();
            check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
            float ms=0; check(cudaEventElapsedTime(&ms,start,end));
            std::printf("attention benchmark n=%d columns=%d path=%s gpu_ms=%.6f\n",n,columns,fused?"fused":"legacy",ms/3);
            check(cudaMemcpyAsync((fused?result:reference).data(),out,result.size()*4,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream)); cudaEventDestroy(start); cudaEventDestroy(end);
        }
        double error=0;
        for(size_t i=0;i<result.size();++i) error=std::max(error,std::abs(double(result[i])-reference[i]));
        std::printf("attention benchmark parity n=%d max_abs_diff=%.9g\n",n,error);
        if(error>=1e-5) throw std::runtime_error("fused attention benchmark parity");
        cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(partial);cudaFree(accum);cudaFree(out);
    }
    check(cudaStreamDestroy(stream));
}
int main(int argc, char** argv) {
    try {
#ifdef LAMINA_PREFILL_BLAS
        matrix_check();
#endif
        if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmark(); return 0; }
        constexpr int heads = 16, kv_heads = 2, dim = 256, stride = 512;
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        for (bool half : {false, true}) {
        for (int n : {1, 127, 128, 129, 513, 2049, 131072}) {
            const int tiles = (n + 127) / 128;
            std::vector<float> q(heads * 2 * dim), k(size_t(n) * stride), v(k.size()), got(heads * dim);
            const bool distant = n == 131072;
            for (size_t i = 0; i < q.size(); ++i) q[i] = distant ? 0.0f : 0.17f * std::sin(float(i) * 0.43f);
            for (size_t i = 0; i < k.size(); ++i) {
                k[i] = distant ? 0.0f : 0.21f * std::cos(float(i) * 0.012f);
                v[i] = distant ? 0.0f : std::sin(float(i) * 0.017f);
            }
            if (distant) for (int j = 0; j < stride; ++j) v[j] = float((half ? 8192 : n) * 2 * ((j % dim) % 7 - 3));
            if (half) {
                for (auto& x : k) x = __half2float(__float2half_rn(x));
                for (auto& x : v) x = __half2float(__float2half_rn(x));
            }
            float *dq, *dk, *dv, *dp, *out;
            void *hk = nullptr, *hv = nullptr;
            check(cudaMalloc(&dq, q.size() * sizeof(float))); check(cudaMalloc(&dk, k.size() * sizeof(float)));
            check(cudaMalloc(&dv, v.size() * sizeof(float))); check(cudaMalloc(&out, got.size() * sizeof(float)));
            check(cudaMalloc(&dp, size_t(heads) * tiles * (dim + 2) * sizeof(float)));
            check(cudaMemcpyAsync(dq, q.data(), q.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(dk, k.data(), k.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(dv, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
            if (half) {
                check(cudaMalloc(&hk, k.size() * sizeof(__half))); check(cudaMalloc(&hv, v.size() * sizeof(__half)));
                lamina::model::cuda::kv_store(dk, hk, int(k.size()), true, stream);
                lamina::model::cuda::kv_store(dv, hv, int(v.size()), true, stream);
            }
            // Match the host-cache submission pattern, including an incomplete final tile.
            for (int begin = 0; begin < n; begin += 2048) {
                const int count = std::min(2048, n - begin);
                if (half) lamina::model::cuda::attn_half_partials(dq,
                    static_cast<const __half*>(hk) + size_t(begin)*stride,
                    static_cast<const __half*>(hv) + size_t(begin)*stride,
                    count, begin/128, tiles, dp, stream);
                else lamina::model::cuda::attn_partials(dq, dk + size_t(begin)*stride, dv + size_t(begin)*stride,
                    count, begin/128, tiles, heads, kv_heads, dim, 1.0f/16, dp, stream);
            }
            lamina::model::cuda::attn_merge(dq, dp, tiles, heads, dim, out, stream);
            check(cudaMemcpyAsync(got.data(), out, got.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            double error = 0;
            for (int h = 0; h < heads; ++h) {
                std::vector<double> scores(n);
                double maximum = -INFINITY, sum = 0;
                if (!distant) {
                    for (int t = 0; t < n; ++t) {
                        double dot = 0;
                        for (int j = 0; j < dim; ++j) dot += double(q[h*2*dim+j]) * k[size_t(t)*stride+(h/8)*dim+j];
                        maximum = std::max(maximum, scores[t] = dot / 16);
                    }
                    for (auto& score : scores) sum += score = std::exp(score-maximum);
                }
                for (int j = 0; j < dim; ++j) {
                    double want = double(v[(h/8)*dim+j]) / (2*n);
                    if (!distant) {
                        double acc = 0;
                        for (int t = 0; t < n; ++t) acc += scores[t] * v[size_t(t)*stride+(h/8)*dim+j];
                        want = (acc / sum) / (1 + std::exp(-double(q[h*2*dim+dim+j])));
                    }
                    error = std::max(error, std::abs(want - got[h*dim+j]));
                }
            }
            std::printf("attention type=%s n=%d max_abs_diff=%.9g\n", half ? "f16" : "f32", n, error);
            if (error >= 1e-5) throw std::runtime_error("tiled attention parity failed");
            if (n >= 129) {
                constexpr int columns = 9;
                std::vector<float> queries(q.size() * columns), column_result(got.size() * columns);
                for (int c = 0; c < columns; ++c) std::copy(q.begin(), q.end(), queries.begin() + c * q.size());
                float *dqcols, *partials, *accumulator, *colout;
                check(cudaMalloc(&dqcols, queries.size() * sizeof(float)));
                check(cudaMalloc(&partials, size_t(columns) * heads * 16 * 258 * sizeof(float)));
                check(cudaMalloc(&accumulator, size_t(columns) * heads * 258 * sizeof(float)));
                check(cudaMalloc(&colout, column_result.size() * sizeof(float)));
                check(cudaMemcpyAsync(dqcols, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                for (int begin = 0; begin < n; begin += 2048)
                    lamina::model::cuda::attn_fused_columns(dqcols,
                        half ? static_cast<const void*>(static_cast<const __half*>(hk)+size_t(begin)*stride) : dk+size_t(begin)*stride,
                        half ? static_cast<const void*>(static_cast<const __half*>(hv)+size_t(begin)*stride) : dv+size_t(begin)*stride,
                        std::min(2048, n-begin), begin, n-columns, columns, accumulator, begin == 0, half, stream);
                lamina::model::cuda::attn_columns_finish(dqcols, accumulator, columns, colout, stream);
                check(cudaMemcpyAsync(column_result.data(), colout, column_result.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
                check(cudaStreamSynchronize(stream));
                double column_error = 0;
                for (int c = 0; c < columns; ++c) {
                    const int prefix = n-columns+c+1;
                    for (int h = 0; h < heads; ++h) {
                        std::vector<double> probabilities(prefix); double maximum = -INFINITY, denominator = 0;
                        if (!distant) {
                            for (int t = 0; t < prefix; ++t) {
                                double score = 0;
                                for (int j = 0; j < dim; ++j) score += double(q[h*512+j]) * k[size_t(t)*stride+(h/8)*dim+j];
                                maximum = std::max(maximum, probabilities[t] = score/16);
                            }
                            for (auto& p : probabilities) denominator += p = std::exp(p-maximum);
                        }
                        for (int j = 0; j < dim; ++j) {
                            double want = double(v[(h/8)*dim+j]) / (2*prefix);
                            if (!distant) {
                                double sum = 0;
                                for (int t = 0; t < prefix; ++t) sum += probabilities[t] * v[size_t(t)*stride+(h/8)*dim+j];
                                want = sum/denominator/(1+std::exp(-double(q[h*512+256+j])));
                            }
                            column_error = std::max(column_error, std::abs(want - column_result[(size_t(c)*heads+h)*dim+j]));
                        }
                    }
                }
                std::printf("causal columns type=%s n=%d max_abs_diff=%.9g\n", half ? "f16" : "f32", n, column_error);
                if (column_error >= 1e-5) throw std::runtime_error("causal column attention parity failed");
                cudaFree(dqcols); cudaFree(partials); cudaFree(accumulator); cudaFree(colout);
            }
            cudaFree(hk); cudaFree(hv);
            cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dp); cudaFree(out);
        }
        }
        check(cudaStreamDestroy(stream)); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
