#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>
#include<math.h>

void InitArray(int *array, int length) {
    for(int i=0; i<length; i++) {
        array[i] = rand() % 100;
    }
}

__global__ void computeSquares(const int *a, const int *b, int *sq_diff, int length) {
    //global id = block id in grid * total threads in a block + thread id within a block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        int diff = a[idx] - b[idx];
        sq_diff[idx] = diff * diff;
    }
}

__global__ void reduce(int *input, int *output, int length) {
    extern __shared__ int shared_mem[];
    int thread_id = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < length) {
        shared_mem[thread_id] = input[idx]; 
    } else {
        shared_mem[thread_id] = 0;
    }
    __syncthreads();

    for(int x=blockDim.x/2; x > 0; x /=2) {
        if(thread_id < x) {
            shared_mem[thread_id] += shared_mem[thread_id + x];
        }

        __syncthreads();
    }

    //final result for the block sum from shared_mem[0] copied to output
    if(thread_id == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

int main() {
    const int length = 100;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (length + threadsPerBlock - 1)/threadsPerBlock;

    // host arrays
    int *h_a = (int*)malloc(length * sizeof(int));
    int *h_b = (int*)malloc(length * sizeof(int));

    // device arrays
    int *d_a;
    int *d_b;
    int *d_result;
    cudaMalloc((void**)&d_a, length * sizeof(int));
    cudaMalloc((void**)&d_b, length * sizeof(int));
    cudaMalloc((void**)&d_result, length * sizeof(int));

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
    
    //Initialize host arrays
    InitArray(h_a, length);
    InitArray(h_b, length);

    //Copy host input arrays to device
    cudaMemcpy(d_a, h_a, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, length * sizeof(int), cudaMemcpyHostToDevice);

    //calling CUDA kernel
    computeSquares<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, length);
    cudaDeviceSynchronize();

    //Parallel reduce
    int *d_partial;
    cudaMalloc((void**)&d_partial, blocksPerGrid * sizeof(int));
    reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_result, d_partial, length);
    cudaDeviceSynchronize();

    int *h_partial = (int*)malloc(blocksPerGrid * sizeof(int));
    cudaMemcpy(h_partial, d_partial, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    int total = 0;
    for(int i=0; i<blocksPerGrid; i++) {
        total += h_partial[i];
    }

    float euclidean_dist = sqrt(total);
    printf("Euclidean distance: %.6f\n", euclidean_dist);

    free(h_a);
    free(h_b);
    free(h_partial);
    free(d_a);
    free(d_b);
    free(d_partial);
    free(d_result);

    return 0;
}

//Compile: nvcc euclidean.cu -o euclidean
//Execute: ./euclidean
