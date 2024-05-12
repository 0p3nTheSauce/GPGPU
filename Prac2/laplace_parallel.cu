/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  Copyright John Urbanic, PSC 2017
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#define COLUMNS    30//126//10
#define ROWS       30//94//10

#ifndef MAX_ITER
#define MAX_ITER 100
#endif

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

double Temperature[ROWS+2][COLUMNS+2];      // temperature grid
double Temperature_last[ROWS+2][COLUMNS+2]; // temperature grid from last iteration
//Added by me
double Temp_Temperature[ROWS+2][COLUMNS+2]; //used to temporarily store the values of Temperature


//   helper routines
void initialize();
void track_progress(int iter);
// Added by me
void printMatrix(double *matrix, int rows, int cols);
void printMatrixSubset(double *matrix, int rows, int cols,
                     int fromRow, int toRow,int fromCol, int toCol);
void setTo(double *matrix, int rows, int cols, double val);
void setToInc(double *matrix, int rows, int cols);
void laplace(double *dt, int *iteration);
int checkResult();
//Kernel prototypes
__global__ void avn_tmpchng(double *Temp, double *Temp_last, int rows, int cols, double *dts);

int main(int argc, char *argv[]) {

    // int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
    struct timeval start_time, stop_time, elapsed_time;  // timers

    max_iterations = MAX_ITER;

    gettimeofday(&start_time,NULL); // Unix timer

    //malloc device
    double *d_Temp, *d_Temp_last, *d_dts;
    int rows = ROWS+2, cols = COLUMNS+2;
    int nelems = rows*cols;
    int nBytes = nelems*sizeof(double);
    checkCudaErrors(cudaMalloc((void **)&d_Temp, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_Temp_last, nBytes));
    //dts could be ROWS * COLS but for simpliity 1-to-1 correspondence with Temp and Temp_last
    checkCudaErrors(cudaMalloc((void **)&d_dts, nBytes)); 
    //malloc host dts
    double *h_dts;
    h_dts = (double *)malloc(nBytes);

    //for printing 
    int fromRow = 22;//86;
    int toRow = 32;//96;
    int fromCol = 22;//118;
    int toCol = 32;//128;
    initialize();                   // initialize Temp_last including boundary conditions
    printf("Temperature after initialization: ");
    printMatrixSubset(*Temperature, rows, cols, fromRow, toRow, fromCol, toCol);
    //setTo(*Temperature_last, rows, cols, 1.0);
    // printf("Temperature_last after initialization: ");
    // printMatrixSubset(*Temperature_last, rows, cols, fromRow, toRow, fromCol, toCol);
    //Transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_Temp, Temperature, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Temp_last, Temperature_last, nBytes, cudaMemcpyHostToDevice));
    
    //setup kernel
    dim3 block(32, 32); //for testing, to change later
    dim3 grid(3, 4);

    //test if kernel working 
    //max_iterations = 1;

    // do until error is minimal or until max steps
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {


        checkCudaErrors(cudaDeviceSynchronize());
        // main calculation: average my four neighbors    
        avn_tmpchng<<<grid, block>>>(d_Temp, d_Temp_last, rows, cols, d_dts);
        checkCudaErrors(cudaGetLastError());
        
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(Temperature, d_Temp, nBytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(Temperature_last, d_Temp_last, nBytes, cudaMemcpyDeviceToHost));
        // printf("Temperature after kernel: ");
        // printMatrixSubset(*Temperature, rows, cols, fromRow, toRow, fromCol, toCol);
        // printf("Temperature_last after kernel: ");
        // printMatrixSubset(*Temperature_last, rows, cols, fromRow, toRow, fromCol, toCol);

        dt = 0.0; // reset largest temperature change
        //checkCudaErrors(cudaMemset(d_dts, 0, nBytes));
        
        checkCudaErrors(cudaDeviceSynchronize());
        //copy dts to host
        checkCudaErrors(cudaMemcpy(h_dts, d_dts, nBytes, cudaMemcpyDeviceToHost));
        //find dt
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < nBytes; i++) {
            dt = fmax(h_dts[i], dt);
        }


        //periodically print test values
        if((iteration % 100) == 0) {
            checkCudaErrors(cudaMemcpy(Temperature, d_Temp, nBytes, cudaMemcpyDeviceToHost));
 	        track_progress(iteration); 
        }

	iteration++;
    }

    //copy results back to host
    checkCudaErrors(cudaMemcpy(Temperature, d_Temp, nBytes, cudaMemcpyDeviceToHost));
    
    // checkCudaErrors(cudaMemcpy(Temperature_last, d_Temp_last, nBytes, cudaMemcpyDeviceToHost));
    printf("Temperature after laplace: ");
    printMatrixSubset(*Temperature, rows, cols, fromRow, toRow, fromCol, toCol);
    printf("Check result: \n");
    if (checkResult()){
        printf("Results correct\n");
    } else {
        printf("Results incorrect");
    }
    // printf("Temperature_last after laplace: ");
    // printMatrixSubset(*Temperature_last, rows, cols, fromRow, toRow, fromCol, toCol);
    gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine

    printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    //Deallocate memory
    checkCudaErrors(cudaFree(d_Temp));
    checkCudaErrors(cudaFree(d_Temp_last));
    checkCudaErrors(cudaFree(d_dts));
    free(h_dts);
    //reset device
    checkCudaErrors(cudaDeviceReset());

    exit(0);
}

__global__ void avn_tmpchng(double *Temp, double *Temp_last, int rows, int cols,
                            double *dts)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * cols + ix;
    double dt = 0;
    if (ix > 0 && ix < cols-1 && iy > 0 && iy < rows-1) 
    {
        Temp[idx] = 0.25 * (Temp_last[idx+1] + Temp_last[idx-1] +
                                    Temp_last[idx+cols] + Temp_last[idx-cols]);
        dt = fmax(fabs(Temp[idx] - Temp_last[idx]), dt);
        Temp_last[idx] = Temp[idx];
    }
    dts[idx] = dt;
}

//laplace algorithm as a function
void laplace(double *dt, int *iteration) {
    //create local variables
    int max_iterations = MAX_ITER;
    double local_dt = *dt;
    int local_iteration = *iteration;
    int i, j;
    //laplace algorithm
    // do until error is minimal or until max steps
    while ( local_dt > MAX_TEMP_ERROR && local_iteration <= max_iterations ) {

        // main calculation: average my four neighbors    
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        
        local_dt = 0.0; // reset largest temperature change

        // copy grid to old grid for next iteration and find latest dt
        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	      local_dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), local_dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }
        
        // periodically print test values
        if((local_iteration % 100) == 0) {
 	    track_progress(local_iteration);
        }

	local_iteration++;
    }
    //pass dt and iterations back to main
    *dt = local_dt;
    *iteration = local_iteration;
}

//check that the output is correct
int checkResult(){
    int iteration=1;                                     // current iteration
    double dt=100;
    int nBytes = (ROWS+2) * (COLUMNS+2) * sizeof(double);
    int i, j;
    const double maxErr = 1e-9; // maximum error for floating point comparison
    memcpy(Temp_Temperature, Temperature, nBytes);
    initialize();
    laplace(&dt, &iteration);
    // printMatrix(*Temp_Temperature, ROWS+2, COLUMNS+2);
    for (i = 0; i < ROWS+2; i++){
        for (j = 0; j < COLUMNS+2; j++){
            if (Temp_Temperature[i][j] - Temperature[i][j] > maxErr){
                printf("Temp_Temperature[%d][%d]: %g\n", i, j, Temp_Temperature[i][j]);
                printf("Temperature[%d][%d]: %g\n", i, j, Temperature[i][j]);
                return 0;
            } 

        }
    }
    return 1;
}

// Print the matrix
void printMatrix(double *matrix, int rows, int cols) {
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", *(matrix + i * cols + j));
        }
        printf("\n");
    }
}

// Print a subset of the matrix
void printMatrixSubset(double *matrix, int rows, int cols,
                     int fromRow, int toRow,int fromCol, int toCol) {
    printf("Matrix:\n");
    for (int i = fromRow; i < toRow; i++) {
        for (int j = fromCol; j < toCol; j++) {
            printf("%7.2f ", *(matrix + i * cols + j));
            //printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}


//set all values of a matrix to same values
void setTo(double *matrix, int rows, int cols, double val) {
    int i, j;
    for (i = 0; i < rows; i++){
        for(j=0; j < cols; j++){
            Temperature_last[i][j] = val;
        }
    }
}
//set all values of matrix to incrementing valeus
void setToInc(double *matrix, int rows, int cols) {
    int i, j;
    int val = 0;
    for (i = 0; i < rows; i++){
        for(j=0; j < cols; j++){
            Temperature_last[i][j] = val;
            val++;
        }
    }
}

// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(){

    int i,j;

    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // set left side to 0 and right to a linear increase
    for(i = 0; i <= ROWS+1; i++) {
        Temperature_last[i][0] = 0.0;
        Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*i;
    }
    
    // set top to 0 and bottom to linear increase
    for(j = 0; j <= COLUMNS+1; j++) {
        Temperature_last[0][j] = 0.0;
        Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
    }
}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = ROWS-5; i <= ROWS; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}

