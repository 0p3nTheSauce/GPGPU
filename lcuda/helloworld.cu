#include <stdio.h>
#include "cuda_runtime.h"

__global__ void hellofromGPU(void)
{printf("Hello world form GPU!\n");}

int main(void)
{
    printf("Hello world from CPU!\n");
    hellofromGPU<<<1,10>>>();
    cudaDeviceReset(); //resets device
    return 0;
}