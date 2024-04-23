// Hands-on Lab 2 for linux 
// Compile: nvcc ... -I \usr\etc summatrix.cu -o summ
// NB: check your own path for the common\inc directory to include as -I 
// Run: ./summ

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_cuda.h>

/*
 * This example implements matrix element-wise addition on the host and GPU.
 * sumMatrixOnHost iterates over the rows and columns of each matrix, adding
 * elements from A and B together and storing the results in C. The current
 * offset in each matrix is stored using pointer arithmetic. sumMatrixOnGPU2D
 * implements the same logic, but using CUDA threads to process each matrix.
 */

void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
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

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

// grid 1D block 2D
__global__ void sumMatrixOnGPU1D(float *A, float *B, float *C, int NX, int NY, int incFac)
{   //adjusted for 1D grid
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = threadIdx.y;
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;

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

dim3 calculateGridSize(int NX, int NY, int blockX, int blockY, int incFac)
{   //ensure that grid size is rounded up:
    //  e.g. if Fgrid is 6.25, Igrid would have been 6, grid becomes 7
    float nx = NX;
    float ny = NY;
    float blockx = blockX;
    float blocky = blockY;
    //decrease tge number of blocks to decrease the number of threads 
    //decrease by incFac x 
    float Fgrid = ((nx * ny) / incFac) / (blocky * blocky);
    int Igrid = ((NX * NY) / incFac) / (blockX * blockY);
    if (Fgrid > Igrid)
    {
        Igrid++; 
    }
    dim3 grid(Igrid);
    return grid; 
}



int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    checkCudaErrors(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 12;  // 14
    int ny = 1 << 12;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int incFac = 16; //increase work of each thread by factor of...

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost (h_A, h_B, hostRef, nx, ny);

	// malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    checkCudaErrors(cudaMalloc((void **)&d_MatA, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatB, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatC, nBytes));

   // setup kernel launch parameters
    int dimx = 64; // default block size if no runtime parameters given
    int dimy = 2;

    if(argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
	
	dim3 block(dimx, dimy);
    dim3 grid = calculateGridSize(nx, ny, block.x, block.y, incFac);

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// initialise CUDA timing
	float milli;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);  // start timing

	// execute the kernel
    checkCudaErrors(cudaDeviceSynchronize());
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny, incFac);
    cudaEventRecord(stop);
	checkCudaErrors(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milli, start, stop);  // time random generation

	printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> (ms): %f \n", grid.x, grid.y,
           block.x, block.y, milli);

    checkCudaErrors(cudaGetLastError());

    // copy kernel result back to host side
    checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // checkCudaErrors device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    checkCudaErrors(cudaFree(d_MatA));
    checkCudaErrors(cudaFree(d_MatB));
    checkCudaErrors(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());

    return EXIT_SUCCESS;
}
