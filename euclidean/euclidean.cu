#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include<math.h>

void InitArray(float *array, int length) {
    for(int i = 0; i < length; i++) {
        array[i] = static_cast<float>(rand() % 100);
    }
}

double serialEuclidean(const float *a, const float *b, int length) {
    double total = 0.0f;

    for(int i=0; i<length; i++) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        total += diff * diff;
    }

    return static_cast<double>(sqrt(total));
}

__global__ void computeSquares(const float *a, const float *b, float *sq_diff, int length) {
    //global id = block id in grid * total threads in a block + thread id within a block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        float diff = a[idx] - b[idx];
        sq_diff[idx] = diff * diff;
    }
}

__global__ void reduce(float *input, float *output, int length) {
    extern __shared__ float shared_mem[];
    int thread_id = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        shared_mem[thread_id] = input[idx];
    } else {
        shared_mem[thread_id] = 0.0f;
    }
    __syncthreads();

    for (int x = blockDim.x / 2; x > 0; x /= 2) {
        if (thread_id < x) {
            shared_mem[thread_id] += shared_mem[thread_id + x];
        }
        __syncthreads();
    }

    //final result for the block sum from shared_mem[0] copied to output
    if (thread_id == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Length of array required!!\n");
        return -1;
    }
    const int length = atoi(argv[1]);
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (length + threadsPerBlock - 1)/threadsPerBlock;

    // host arrays
    float *h_a = (float*)malloc(length * sizeof(float));
    float *h_b = (float*)malloc(length * sizeof(float));

    // device arrays
    float *d_a;
    float *d_b;
    float *d_result;
    cudaMalloc((void**)&d_a, length * sizeof(float));
    cudaMalloc((void**)&d_b, length * sizeof(float));
    cudaMalloc((void**)&d_result, length * sizeof(float));

    cudaEvent_t cuda_start;
    cudaEvent_t cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    //print cuda enabled hardware devices
    int cuda_devices;
    cudaError_t err_code = cudaGetDeviceCount(&cuda_devices);

    if (cuda_devices < 1) {
        printf("WAIT A MINUTE.. There are no CUDA enabled devices!!!\n");
        return -1;
    }

    if(err_code != cudaSuccess) {
        printf("Error while getting CUDA enabled hardware device count!");
    }
    else {
        printf("Number of cuda enabled devices: %d\n", cuda_devices);
    }

    //Device 0 properties
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    printf("Device name: %s\n", device_properties.name);
    printf("Prop 1: Max threads per block: %d\n", device_properties.maxThreadsPerBlock);
    printf("Prop 2: Registers per block: %d\n", device_properties.regsPerBlock);
    printf("Prop 3: Total global memory: %zu\n", device_properties.totalGlobalMem);
    printf("Prop 4: Shared memory per block: %zu\n", device_properties.sharedMemPerBlock);
    
    // Initialize host arrays
    InitArray(h_a, length);
    InitArray(h_b, length);

    // Copy host input arrays to device
    cudaMemcpy(d_a, h_a, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, length * sizeof(float), cudaMemcpyHostToDevice);

    //calling CUDA kernel
    cudaEventRecord(cuda_start);
    computeSquares<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, length);
    cudaDeviceSynchronize();

    // Parallel reduce
    float *d_partial;
    cudaMalloc((void**)&d_partial, blocksPerGrid * sizeof(float));
    reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_result, d_partial, length);
    cudaDeviceSynchronize();

    float *h_partial = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);

    float cuda_time;
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);
    printf("CUDA Time: %.6f s\n", cuda_time/1000.0f);

    float total = 0.0f;
    for(int i=0; i<blocksPerGrid; i++) {
        total += h_partial[i];
    }

    double euclidean_dist = sqrt(total);
    printf("Euclidean distance - CUDA: %.6f\n", euclidean_dist);

    //Serial Euclidean
    clock_t start = clock();
    float serial_euc = serialEuclidean(h_a, h_b, length);
    clock_t end = clock();
    printf("Serial Time: %.6f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Euclidean distance - Serial: %.6f\n", serial_euc);

    free(h_a);
    free(h_b);
    free(h_partial);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
    cudaFree(d_result);

    return 0;
}

//Compile: nvcc euclidean.cu -o euclidean
//Execute: ./euclidean
