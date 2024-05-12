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
//Added by me 
#include <string.h>

// size of plate
#define COLUMNS    10
#define ROWS       10

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
// Created by me
void printMatrix(double *matrix, int rows, int cols);
void laplace(double *dt, int *iteration);
int checkResult();

int main(int argc, char *argv[]) {

    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100;                                       // largest change in t
    struct timeval start_time, stop_time, elapsed_time;  // timers

    max_iterations = MAX_ITER;

    gettimeofday(&start_time,NULL); // Unix timer

    initialize();                   // initialize Temp_last including boundary conditions
    printf("Temperature after initialization: \n");
    printMatrix(*Temperature, ROWS+2,COLUMNS+2 );
    printf("Temperature_last after initialization: \n");
    printMatrix(*Temperature_last, ROWS+2,COLUMNS+2 );
    //printMatrix(*Temperature_last, ROWS+2,COLUMNS+2 );


    // do until error is minimal or until max steps
    while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {

        // main calculation: average my four neighbors    
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        
        dt = 0.0; // reset largest temperature change

        // copy grid to old grid for next iteration and find latest dt
        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	      dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }
        
        // periodically print test values
        if((iteration % 100) == 0) {
 	    track_progress(iteration);
        }

	iteration++;
    }

    //laplace(&dt, &iteration);

    printf("Temperature after laplace: \n");
    printMatrix(*Temperature, ROWS+2,COLUMNS+2 );
    printf("Check result: \n");
    if (checkResult()){
        printf("Results correct\n");
    } else {
        printf("Results incorrect");
    }
    //printMatrix(*Temperature_last, ROWS+2,COLUMNS+2 );
    gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine

    printf("\nMax error at iteration %d was %f\n", iteration-1, dt);
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

    exit(0);
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
    memcpy(Temp_Temperature, Temperature, nBytes);
    initialize();
    laplace(&dt, &iteration);
    // printMatrix(*Temp_Temperature, ROWS+2, COLUMNS+2);
    for (i = 0; i < ROWS+2; i++){
        for (j = 0; j < COLUMNS+2; j++){
            if (Temp_Temperature[i][j] != Temperature[i][j]) return 0;
        }
    }
    return 1;
}

// Function definition to print the matrix
void printMatrix(double *matrix, int rows, int cols) {
    // printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%7.2f ", *(matrix + i * cols + j));
        }
        printf("\n");
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
