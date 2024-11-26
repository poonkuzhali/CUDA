#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include<cuda_runtime.h>

void serial_implementation(unsigned h, unsigned w, unsigned char *h_input) {
    for (unsigned y = 0; y < h; y++) {
        for (unsigned x = 0; x < w; x++) {
            unsigned char* pixel = &h_input[4 * (y * w + x)];
            unsigned char gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
            pixel[0] = pixel[1] = pixel[2] = gray;
        }
    }
}

__global__ void cuda_convertImage(unsigned char *d_input, unsigned char *d_output, unsigned w, unsigned h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < w * h) {
        int pixel_idx = idx * 4;
        unsigned char r = d_input[pixel_idx];
        unsigned char g = d_input[pixel_idx + 1];
        unsigned char b = d_input[pixel_idx + 2];

        unsigned char gray = 0.299 * r + 0.587 * g + 0.114 * b;

        d_output[pixel_idx] = gray;
        d_output[pixel_idx + 1] = gray;
        d_output[pixel_idx + 2] = gray;
        d_output[pixel_idx + 3] = d_input[pixel_idx + 3];

    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Input and output images required!!\n");
        return -1;
    }

    unsigned error;
    unsigned char* h_input = NULL;
    unsigned char* h_output = NULL;
    unsigned w, h;

    //Decoding input
    error = lodepng_decode32_file(&h_input, &w, &h, argv[1]);
    if (error) {
        printf("Error decoding image: %s\n", lodepng_error_text(error));
        return 1;
    }

    clock_t start = clock();
    serial_implementation(h, w, h_input);
    clock_t end = clock();

    printf("Serial time: %.6f s", (double)(end-start)/CLOCKS_PER_SEC);

    error = lodepng_encode32_file("CPU_grayscale.png", h_input, w, h);
    if (error) {
        printf("Error encoding image: %s\n", lodepng_error_text(error));
        free(h_input);
        return 1;
    }

    //allocate memory on device
    size_t size = w * h * 4;
    unsigned char *d_input = NULL;
    unsigned char *d_output = NULL;

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    error = lodepng_decode32_file(&h_input, &w, &h, argv[1]);

    if (error) {
        printf("Error decoding image: %s\n", lodepng_error_text(error));
        return 1;
    }

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (w * h + threadsPerBlock - 1)/threadsPerBlock;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cuda_convertImage<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, w, h);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("CUDA Time: %.6f s\n", cuda_time/1000.0f);


    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    printf("Image converted to grayscale successfully.\n");
    return 0;
}