#include <stdio.h>
#include <stdlib.h>
#include<cuda_runtime.h>
#include<time.h>

void InitMatrix(float *matrix, int N) {
    for (int i = 0; i < N*N; i++) {
        matrix[i] = (float)(rand() % 100)/10.0f;
    }
}

void serialMultiplication(float *a, float *b, float *result, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = 0;
            for (int k = 0; k < N; k++) {
                val += a[i * N + k] * b[k * N + j];
            }
            result[i * N + j] = val;
        }
    }
}

__global__ void gpuMultiplication(float *a, float *b, float *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0;
        for (int k = 0; k < N; k++) {
            val += a[row * N + k] * b[k * N + col];
        }
        result[row * N + col] = val;
    }
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Matrix size required!!\n");
        return -1;
    }

    const int N = atoi(argv[1]);
    printf("Matrix size: %d\n", N);
    size_t size = N * N * sizeof(float);

    //host matrices
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_cpu = (float*)malloc(size);
    float *h_gpu = (float*)malloc(size);

    //Initialize matrices
    InitMatrix(h_a, N);
    InitMatrix(h_b, N);
    printf("Matrix initialized!\n");

    //CPU
    clock_t start = clock();
    serialMultiplication(h_a, h_b, h_cpu, N);
    clock_t end = clock();
    printf("CPU Time: %.6f s\n", (double)(end - start) / CLOCKS_PER_SEC);

    //device matrices
    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    //copy host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = atoi(argv[2]);
    dim3 threadsPerBlock(threads, threads);
    dim3 blocksPerGrid((N+threads-1)/threads, (N+threads-1)/threads);

    cudaEvent_t cuda_start;
    cudaEvent_t cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    cudaEventRecord(cuda_start);
    gpuMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);

    float cuda_time;
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);
    printf("CUDA Time: %.6f s\n", cuda_time/1000.0f);

    cudaMemcpy(h_gpu, d_c, size, cudaMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_cpu[i] - h_gpu[i]) > 1e-4) {
            correct = 0;
            break;
        }
    }
    printf("Verification: %s\n", correct ? "SUCCESS" : "FAILED");

    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
