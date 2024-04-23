#include <stdio.h>
#include "cuda_runtime.h"
#include "helper_cuda.h"

__global__ void vecAdd(float *A, float *B, float *C, const int N)
{
    unsigned int i = threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void sumArraysOnCPU(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
    return;
}

// Compute vector sum C = A + B
void sumArraysOnGPU(float *h_A, float *h_B, float *h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A,*d_B,*d_C;

    //allocate space on GPU
    checkCudaErrors(cudaMalloc((void **) &d_A, size));
    //cudaMalloc((void **) &d_A, size); 
    checkCudaErrors(cudaMalloc((void **) &d_B, size));
    checkCudaErrors(cudaMalloc((void **) &d_C, size));
    //copy vectors to GPU
    checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    //Kernel invocation
    vecAdd<<<1, n>>>(d_A, d_B, d_C, n);
    //copy array back to CPU
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    //deallocate memory
    checkCudaErrors(cudaFree(d_A)); checkCudaErrors(cudaFree(d_B)); checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaDeviceReset());
}

int main()
{
    //I/O to read N
    int nElem = 0;
    printf("Please enter size array: ");
    scanf("%d", &nElem);
    while (nElem < 3) 
    {
        printf("Too small, make array at least size 3: ");
        scanf("%d", &nElem);
    }
    // or set nElem to be 1000, #define nElem = 1000


    //Memory allocation for h_A, h_B, and h_C
    int size = nElem * sizeof(float);
    float *h_A, *h_B, *h_C, *d_C;
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    d_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL){
        printf("Malloc failed\n");
        return 1;
    }
    // create arrays
    for (int i = 0; i < nElem; i++)
    {
        h_A[i] = i;
        h_B[i] = i;
    }

    //add arrays
    sumArraysOnCPU(h_A, h_B, h_C, nElem);
    sumArraysOnGPU(h_A, h_B, d_C, nElem);
    //output results
    printf("Vector A: [%f", h_A[(nElem-3)]); //only show the last 3 elements
    for (int i = (nElem-2); i < nElem; i++)
    {
        printf(", %f", h_A[i]);
    }
    printf("]\n");
    printf("Vector B: [%f", h_B[(nElem-3)]);//only show the last 3 elements
    for (int i = (nElem-2); i < nElem; i++)
    {
        printf(", %f", h_B[i]);
    }
    printf("]\n");
    printf("Host vector C: [%f", h_C[(nElem-3)]);//only show the last 3 elements
    for (int i = (nElem-2); i < nElem; i++)
    {
        printf(", %f", h_C[i]);
    }
    printf("]\n");
    printf("Device vector C: [%f", d_C[(nElem-3)]);//only show the last 3 elements
    for (int i = (nElem-2); i < nElem; i++)
    {
        printf(", %f", d_C[i]);
    }
    printf("]\n");
    //deallocate memory
    free(h_A);free(h_B);free(h_C);free(d_C);
    return EXIT_SUCCESS;
}
