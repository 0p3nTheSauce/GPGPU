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
    int id = threadIdx.x + blockIdx.x; // * 64
    /*  Each thread gets same seed, 
        a different sequence number, no offset
    */
   curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int *result)
{
    int id = threadIdx.x + blockIdx.x; // * 64
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

}