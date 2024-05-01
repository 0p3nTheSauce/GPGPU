/*  Program to compute Pi using Monte Carlo methods, on the GPU
    Compile with nvcc PMCPi.cu -lcurand -o PMCPi
    Run with MCPi
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

__global__ void generate_kernel(curandState *state, int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x; 
    int count = 0;
    double x, y, z;

    //copy state to local memory for efficiency
    curandState localState = state[id];

    //generate pseudo-random unsigned int
    for (int n = 0; n < 8; n++) {
        /*  we want our x and y to fall in the range -1 to 1.
            curand_uniform returns values in a uniform distribution,
            in the range 0 to 1, (annoyingly) excluding 0 but including 1.
            I'm going to pretend that it returns values in the range
            0 to 1 inclusive, so that i can modify the range to -1 to 1
        */
        x = curand_uniform(&localState);
        x = x * 2;
        x = x - 1;
        y = curand_uniform(&localState);
        y = y * 2;
        y = y - 1;
        z = x*x + y*y;
        if (z<=1) count++;
    }

    //Copy state back to global memory
    state[id] = localState;

    //Store results
    result[id] += count;
}

int main(int argc, char **argv)
{
    int niter=0;
    double pi;
    curandState *devStates;
    int *devResults, *hostResults;
    int count = 0;

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

    /*  Allocate space for results on host
        8 calculations per thread * 2048 threads per SM * 6 SMs
        = 98304 iterations
        98304 iterations / 8 results per thread = 12288 results
        (probably a good area for optimization later)
    */
    hostResults = (int *)calloc(12288, sizeof(int));

    //Allocate space for results on device 
    checkCudaErrors(cudaMalloc((void **)&devResults,
                    12288 * sizeof(int)));

    //Set results to 0
    checkCudaErrors(cudaMemset(devResults, 0,
                    12288 * sizeof(int)));

    //Allocate space for prng states on device
    checkCudaErrors(cudaMalloc((void **)&devStates,
                    12288 * sizeof(curandState)));
    
    /*  Setup prng states
        2048 threads per SM = 32 blocks, 64 threads each 
        32 blocks * 6 SMs = 192 blocks total 
    */
    setup_kernel<<<192, 64>>>(devStates);
    
    //Generate pseudo-random
    generate_kernel<<<192, 64>>>(devStates, devResults);

    //Copy device memory to host
    checkCudaErrors(cudaMemcpy(hostResults, devResults,
                    12288 * sizeof(int), cudaMemcpyDeviceToHost));

    //Calculate total count
    for (int i = 0; i < 12288; i++) {
        count += hostResults[i];
    }
    //calculate pi
    pi = (double)count/98304*4; //I will have to change this at some point
    //pi=(double)count/niter*4;
    printf("# of trials= %d , estimate of pi is %g \n",98304,pi);
    return 0;

}