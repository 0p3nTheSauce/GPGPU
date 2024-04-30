#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

/*  A Device API example program using the CURAND library
    which generates 100000 pseudorandom floats,
    and checks what fraction have the lower bit set (are odd)
    taken from the slide which are taken form the cuda toolkit documentation. 

    compile with: nvcc Ex_CURAND_device_API.cu -lcurand -o ExdeviceAPI
*/

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /*  Each thread gets same seed, 
        a different sequence number, no offset
    */
   curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    unsigned int x;

    /*Copy state to local memory for efficiency */
    curandState localState = state[id];

    /* Generate pseudo-random unsigned ints */
    for (int n = 0; n < 100000; n++) {
        x = curand(&localState);
        /* Check if low bit set */
        if (x & 1) count++;
    }

    /* Copy state back to global memoryt */
    state[id] = localState;

    /* Store results*/
    result[id] += count;
}

int main()
{
    int i, total;
    curandState *devStates;
    int *devResults, *hostResults;

    /* Allocate space for results on host*/
    hostResults = (int *)calloc(64 * 64, sizeof(int));

    /* Allocate spce for results on device*/
    CUDA_CALL(cudaMalloc((void **)&devResults, 64 * 64 * sizeof(int)));

    /* Set results to 0*/
    CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * sizeof(int)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState)));

    /* Setup prng states*/
    setup_kernel<<<64, 64>>>(devStates);

    /* Generate and use pseudo-random */
    for(i = 0; i < 10; i++) {
        generate_kernel<<<64, 64>>>(devStates, devResults);
    }

    /* copy device memory to host*/
    CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * sizeof(int), cudaMemcpyDeviceToHost));

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    printf("Fraction with low bit set was %10.13f\n",
    (float)total / (64.0f * 64.0f * 100000.0f * 10.0f));
    
    /* cleanup */
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);    
    return 0;
}