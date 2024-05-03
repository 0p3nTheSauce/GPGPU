/*  Program to compute Pi using Monte Carlo methods, on the GPU
    Compile with nvcc PMCPi.cu -lcurand -o PMCPi
    Run with PMCPi
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#define SEED 35791246

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /*  Each thread gets same seed, 
        a different sequence number, no offset
    */
   curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int *result, int calcpt)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x; 
    int count = 0;
    double x, y, z;

    //copy state to local memory for efficiency
    curandState localState = state[id];

    //generate pseudo-random unsigned int
    for (int n = 0; n < calcpt; n++) {
        /*  curand uniform returns pseudo random numbers in the range 0 to 1
            the algorithm still works because we are just calculating for a 
            qurter of a circle
        */
        x = curand_uniform_double(&localState);
        y = curand_uniform_double(&localState);
        z = x*x + y*y;
        if (z<=1) count++;
    }

    //Store results
    result[id] += count;
}

int main(int argc, char **argv)
{
    double pi;
    curandState *devStates;
    int *devResults, *hostResults;
    int count = 0;
    int nt = 2048 * 6; //total number of threads (6 SMs * 2048 threads)
    int niter, calcpt, riter;
    //Take input from command line
    if (argc != 2)
    {
      printf("Usage: PMCPi <number_of_iterations>\n");
      exit(EXIT_FAILURE);
    } 
    niter = atoi(argv[1]);
    if (niter <= 0) 
    {
      printf("Number of iterations must be a positive integer\n");
      exit(EXIT_FAILURE);
    }
    calcpt = niter / nt; //calculations per thread
    riter = nt * calcpt; //rounded iterations

    //Allocate space for results on host
    hostResults = (int *)calloc(nt, sizeof(int));

    //Allocate space for results on device 
    checkCudaErrors(cudaMalloc((void **)&devResults,
                    nt * sizeof(int)));

    //Allocate space for prng states on device
    checkCudaErrors(cudaMalloc((void **)&devStates,
                    nt * sizeof(curandState)));
    
    /*  Setup prng states
        2048 threads per SM = 32 blocks, 64 threads each 
        32 blocks * 6 SMs = 192 blocks total 
        (good place to return for optimisation)
    */
    setup_kernel<<<192, 64>>>(devStates);
    
    //Generate pseudo-random
    generate_kernel<<<192, 64>>>(devStates, devResults, calcpt);

    //Copy device memory to host
    checkCudaErrors(cudaMemcpy(hostResults, devResults,
                    nt * sizeof(int), cudaMemcpyDeviceToHost));

    //Calculate total count
    for (int i = 0; i < nt; i++) {
        count += hostResults[i];
    }
    //calculate pi
    pi = (double)count/riter*4; 
    printf("# of trials= %d , estimate of pi is %g \n",riter,pi);
    
    //clean up
    free(hostResults);
    checkCudaErrors(cudaFree(devResults));
    checkCudaErrors(cudaFree(devStates));
    checkCudaErrors(cudaDeviceReset());
    
    return 0;

}