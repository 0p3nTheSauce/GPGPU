#include <stdio.h>
//#include <cuda.h>
//#include "cuda_runtime.h"
#include "helper_cuda.h"

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Compute vector sum C = A + B
void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A,*d_B,*d_C;

    //allocate space on GPU
    checkCudaErrors(cudaMalloc((void **) &d_A, size));
    //cudaMalloc((void **) &d_A, size); 
    checkCudaErrors(cudaMalloc((void **) &d_B, size));
    checkCudaErrors(cudaMalloc((void **) &d_C, size));
    //copy vectors to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    //Kernel invocation
    sumArraysOnGPU<<<1, n>>>(d_A, d_B, d_C, n);
    //copy array back to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    //deallocate memory
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
    cudaDeviceReset();
}

int main()
{
    //I/O to read N
    int nElem = 0;
    printf("Please enter size array: ");
    scanf("%d", &nElem);
    //Memory allocation for h_A, h_B, and h_C
    int size = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL){
        printf("Malloc failed\n");
        return 1;
    }
    //I/O to read h_A and h_B, N elements
    printf("For vector A: \n");
    for (int i = 0; i < nElem; i++)
    {
        printf("Enter an integer: ");
        scanf("%f", &h_A[i]);
    }
    printf("For vector B: \n");
    for (int i = 0; i < nElem; i++)
    {
        printf("Enter an integer: ");
        scanf("%f", &h_B[i]);
    }
    //add arrays
    vecAdd(h_A, h_B, h_C, nElem);
    //output results
    printf("Vector A: [%f", h_A[0]);
    for (int i = 1; i < nElem; i++)
    {
        printf(", %f", h_A[i]);
    }
    printf("]\n");
    printf("Vector B: [%f", h_B[0]);
    for (int i = 1; i < nElem; i++)
    {
        printf(", %f", h_B[i]);
    }
    printf("]\n");
    printf("Vector C: [%f", h_C[0]);
    for (int i = 1; i < nElem; i++)
    {
        printf(", %f", h_C[i]);
    }
    printf("]\n");
    //deallocate memory
    free(h_A);free(h_B);free(h_C);
}
