#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#define COLUMNS    1000
#define ROWS       1000

#ifndef MAX_ITER
#define MAX_ITER 100
#endif

#define MAX_TEMP_ERROR 0.01

double Temp_Temperature[ROWS+2][COLUMNS+2];

void initialize(double *Temperature, double *Temperature_last);
void track_progress(int iter, double *Temperature);
int checkResult(double *Temperature, double *Temperature_last);

__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void averageNeighbours(double *Temp, double *Temp_last, int rows, int cols) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * cols + ix;

    if (ix > 0 && iy > 0 && ix < cols - 1 && iy < rows - 1) {
        Temp[idx] = 0.25 * (Temp_last[idx - 1] + Temp_last[idx + 1] + Temp_last[idx - cols] + Temp_last[idx + cols]);
    }
}

__global__ void temperatureChange(double *Temp, double *Temp_last, int rows, int cols, double *dts) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * cols + ix;

    if (ix > 0 && iy > 0 && ix < cols - 1 && iy < rows - 1) {
        double dt = fabs(Temp[idx] - Temp_last[idx]);
        atomicMaxDouble(dts, dt);
        Temp_last[idx] = Temp[idx];
    }
}

int main(int argc, char *argv[]) {
    int max_iterations = MAX_ITER;
    int iteration = 1;
    double dt = 100;
    struct timeval start_time, stop_time, elapsed_time;

    gettimeofday(&start_time, NULL);

    int rows = ROWS + 2;
    int cols = COLUMNS + 2;
    int nelems = rows * cols;
    int nBytes = nelems * sizeof(double);

    double *Temperature, *Temperature_last, *dts;
    cudaMallocManaged(&Temperature, nBytes);
    cudaMallocManaged(&Temperature_last, nBytes);
    cudaMallocManaged(&dts, sizeof(double));
    *dts = 0;

    initialize(Temperature, Temperature_last);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {
        averageNeighbours<<<grid, block>>>(Temperature, Temperature_last, rows, cols);
        cudaDeviceSynchronize();

        temperatureChange<<<grid, block>>>(Temperature, Temperature_last, rows, cols, dts);
        cudaDeviceSynchronize();

        dt = *dts;
        *dts = 0;

        if ((iteration % 100) == 0) {
            track_progress(iteration, Temperature);
        }

        iteration++;
    }

    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);

    printf("\nMax error at iteration %d was %f\n", iteration - 1, dt);
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);

    printf("Check result: \n");
    if (checkResult(Temperature, Temperature_last)) {
        printf("Results correct\n");
    } else {
        printf("Results incorrect\n");
    }

    cudaFree(Temperature);
    cudaFree(Temperature_last);
    cudaFree(dts);

    return 0;
}

void initialize(double *Temperature, double *Temperature_last) {
    int i, j;
    for (i = 0; i <= ROWS + 1; i++) {
        for (j = 0; j <= COLUMNS + 1; j++) {
            Temperature_last[i * (COLUMNS + 2) + j] = 0.0;
        }
    }

    for (i = 0; i <= ROWS + 1; i++) {
        Temperature_last[i * (COLUMNS + 2)] = 0.0;
        Temperature_last[i * (COLUMNS + 2) + COLUMNS + 1] = (100.0 / ROWS) * i;
    }

    for (j = 0; j <= COLUMNS + 1; j++) {
        Temperature_last[j] = 0.0;
        Temperature_last[(ROWS + 1) * (COLUMNS + 2) + j] = (100.0 / COLUMNS) * j;
    }
}

void track_progress(int iteration, double *Temperature) {
    int i;
    printf("---------- Iteration number: %d ------------\n", iteration);
    for (i = ROWS - 5; i <= ROWS; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i * (COLUMNS + 2) + i]);
    }
    printf("\n");
}

int checkResult(double *Temperature, double *Temperature_last) {
    int iteration = 1;
    double dt = 100;
    int rows = ROWS + 2;
    int cols = COLUMNS + 2;
    int i, j;
    const double maxErr = 1e-6;
    memcpy(Temp_Temperature, Temperature, rows * cols * sizeof(double));
    initialize(Temperature, Temperature_last);

    while (dt > MAX_TEMP_ERROR && iteration <= MAX_ITER) {
        for (i = 1; i <= ROWS; i++) {
            for (j = 1; j <= COLUMNS; j++) {
                Temperature[i * cols + j] = 0.25 * (Temperature_last[(i + 1) * cols + j] +
                                                    Temperature_last[(i - 1) * cols + j] +
                                                    Temperature_last[i * cols + (j + 1)] +
                                                    Temperature_last[i * cols + (j - 1)]);
            }
        }

        dt = 0.0;
        for (i = 1; i <= ROWS; i++) {
            for (j = 1; j <= COLUMNS; j++) {
                dt = fmax(fabs(Temperature[i * cols + j] - Temperature_last[i * cols + j]), dt);
                Temperature_last[i * cols + j] = Temperature[i * cols + j];
            }
        }

        iteration++;
    }

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (fabs(Temp_Temperature[i][j] - Temperature[i * cols + j]) > maxErr) {
                printf("Temp_Temperature[%d][%d]: %g\n", i, j, Temp_Temperature[i][j]);
                printf("Temperature[%d][%d]: %g\n", i, j, Temperature[i * cols + j]);
                return 0;
            }
        }
    }
    return 1;
}
