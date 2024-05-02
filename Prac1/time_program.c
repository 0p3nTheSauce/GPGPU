#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STRING 100

int main(int argc, char **argv) {
    clock_t start, end;
    double exec_time = 0;
    double total_exec_time = 0;
    char *program;
    char *niter;
    char execute[100] = "./";
    if (argc == 3)
    {
        program = argv[1];
        niter = argv[2];
    } else 
    {
        printf("Usage: time_program <program_to_time> <number_of_iterations>\n");
        exit(EXIT_FAILURE);
    }
    strcat(execute, program);
    strcat(execute, " ");
    strcat(execute, niter);
    // calculate average execution time
    for (int i = 0; i < 10; i++) 
    {
        start = clock();
        system(execute);
        end = clock();
        total_exec_time += ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
    }
    exec_time = total_exec_time / 10;
    printf("Average execution time: %f ms\n", exec_time);
    
    return 0;
}