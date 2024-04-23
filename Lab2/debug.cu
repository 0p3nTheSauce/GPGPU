#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_cuda.h>

dim3 calculateGridSize(int NX, int NY, int blockX, int blockY, int incFac)
{
    //dim3 grid( (nx * ny) / (blockX * blockY));
    float nx = NX;
    float ny = NY;
    float blockx = blockX;
    float blocky = blockY;
    //decrease the number of blocks to decrease the number of threads 
    float Fgrid = ((nx * ny) / incFac) / (blocky * blocky);
    int Igrid = ((NX * NY) / incFac) / (blockX * blockY);
    printf("Fgrid: %f\n", Fgrid);
    printf("Igrid: %d\n", Igrid);
    if (Fgrid > Igrid)
    {
        Igrid++;
    }
    dim3 grid(Igrid);
    printf("Grid: (%d, %d)\n", grid.x, grid.y);
    return grid; 
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY, int incFac)
{
    printf("blockIdx.x: %d\n", blockIdx.x);
    printf("bloackDim.x: %d\n", blockDim.x);
    printf("threadIdx.x: %d\n", threadIdx.x);
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    printf("ix: %d\n", ix);
    unsigned int iy = threadIdx.y;
    printf("iy: %d\n", iy);
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;
    printf("Idx: %d\n", idx);

    //make thread do incFac x work
    unsigned int adjIdx = idx * incFac; //adjusted index because doing incFac x work

    for (int i = 0; i < incFac; i++) //repeat incFac number of times
    {
        if (adjIdx < (NX * NY))
        {
            C[adjIdx] = A[adjIdx] + B[adjIdx];
        }
        adjIdx++;
    }
}


void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void printArray(float *array, int size)
{
    printf("[%f",array[0]);
    for (int i = 1; i < size; i++)
    {
        printf(", %f", array[i]);
    }
    printf("]\n");
    return;
}

int main()
{
    int nx = 5;
    int ny = 5;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    int incFac = 3; //increase work of each thread by factor
    //Host
    float *arrA, *arrB, *arrC, *arrCD; //c = a + b
    arrA = (float *)malloc(nBytes);
    arrB = (float *)malloc(nBytes);
    arrC = (float *)malloc(nBytes);
    arrCD = (float *)malloc(nBytes);
    //Device
    float *dArrA, *dArrB, *dArrC;
    checkCudaErrors(cudaMalloc((void **)&dArrA, nBytes));
    checkCudaErrors(cudaMalloc((void **)&dArrB, nBytes));
    checkCudaErrors(cudaMalloc((void **)&dArrC, nBytes));

    int dimx = 2;
    int dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid = calculateGridSize(nx, ny, block.x, block.y, incFac);

    initialData(arrA, nxy);
    initialData(arrB, nxy);
    memset(arrC, 0, nBytes);
    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(dArrA, arrA, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dArrB, arrB, nBytes, cudaMemcpyHostToDevice));

    sumMatrixOnHost(arrA, arrB, arrC, nx ,ny);
    // execute the kernel
    checkCudaErrors(cudaDeviceSynchronize());
    sumMatrixOnGPU2D<<<grid, block>>>(dArrA, dArrB, dArrC, nx, ny, incFac);
    // copy kernel result back to host side
    checkCudaErrors(cudaMemcpy(arrCD, dArrC, nBytes, cudaMemcpyDeviceToHost));

    printf("Original: \n");
    printArray(arrA, nxy);
    printArray(arrB, nxy);
    printf("On host: \n");
    printArray(arrC, nxy);
    printf("On GPU: \n");
    printArray(arrCD, nxy);


    // free device global memory
    checkCudaErrors(cudaFree(dArrA));
    checkCudaErrors(cudaFree(dArrB));
    checkCudaErrors(cudaFree(dArrC));
    free(arrA);free(arrB);free(arrC);free(arrCD);
    return 0;
}