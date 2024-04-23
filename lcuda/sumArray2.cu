// C version of sumarray from GPU course slides
// Compile: nvcc sumArray.cu -o sumarray
// or if this gives errors nvcc -gencode arch=compute_50,code=sm_50  sumArray.cu  -o sumarray
// Run: sumarray.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
 * This example implements Array addition on the host and GPU.
 * sumArrayOnHost iterates over the elements, adding
 * elements from A and B together and storing the results in C. 
 * sumArrayOnGPU implements the same logic, but using CUDA threads to process each element.
 */


void sumArrayOnHost(float *A, float *B, float *C, const int n)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

     for (int ix = 0; ix < n; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
    return;
}


__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N)
{
    //unsigned int idx = threadIdx.x;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

#define nelem 1000

void checkerror(cudaError err)
{ if (err != cudaSuccess) fprintf(stderr, "%s \n", cudaGetErrorString(err));
	return; 
}


int main(int argc, char **argv)
{
    int nBytes = nelem * sizeof(float);
    int numthread = 32; // set num threads per block 
    int numblock = (nelem + numthread-1)/numthread;
    // or int numblock = (nElem -1) / numthread +1; 

    // malloc host memory
    float *h_A, *h_B, *hostC, *gpuC;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostC = (float *)malloc(nBytes);
    gpuC = (float *)malloc(nBytes);

    // initialise A and B
     for (int i=0; i < nelem; i++)
      { h_A[i] = i;
        h_B[i] = i; }

    // add Array at host side for result checks
    sumArrayOnHost (h_A, h_B, hostC, nelem);
    printf("Host sum is: %f \n", hostC[9]);

	// malloc device global memory
    float *d_A , *d_B, *d_C;
    checkerror(cudaMalloc((void **)&d_A , nBytes)); 
    checkerror(cudaMalloc((void **)&d_B, nBytes)); 
    checkerror(cudaMalloc((void **)&d_C, nBytes)); 

     // transfer data from host to device
    checkerror(cudaMemcpy(d_A , h_A, nBytes, cudaMemcpyHostToDevice));
    checkerror(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

	// execute the kernel
    checkerror(cudaDeviceSynchronize());
    //sumArrayOnGPU<<<1,nelem>>>(d_A, d_B, d_C, nelem);
    sumArrayOnGPU<<<numblock,numthread>>>(d_A, d_B, d_C, nelem);
    checkerror(cudaGetLastError());

 
    // copy kernel result back to host side
    checkerror(cudaMemcpy(gpuC, d_C, nBytes, cudaMemcpyDeviceToHost));
    printf("GPU sum is: %f \n", gpuC[9]);

     // free device global memory
    checkerror(cudaFree(d_A ));
    checkerror(cudaFree(d_B));
    checkerror(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostC);
    free(gpuC);

    // reset device
    checkerror(cudaDeviceReset());

    return EXIT_SUCCESS;
}
