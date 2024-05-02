#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_STRING 100

/*  This porgram calculates the average execution time of other programs.
    Example: ./time_program 10 MCPi 100000000 
    finds the average execution time of MCPi over 10 runs, with 100000000 as 
    the argument for MCPi
*/

int main(int argc, char **argv) {
    clock_t start, end;
    double exec_time = 0;
    double total_exec_time = 0;
    char *program;
    int niter;
    char execute[100] = "./";
    if (argc < 3)
    {
        printf("Usage: time_program <number_of_iterations> <program_to_time> {<additional_arguments} \n");
        exit(EXIT_FAILURE);
    }
    niter = atoi(argv[1]);
    program = argv[2];
    strcat(execute, program);
    for (int i = 3; i < argc; i++){
        strcat(execute, " ");
        strcat(execute, argv[i]);
    }
    
    // calculate average execution time
    for (int i = 0; i < niter; i++) 
    {
        start = time(NULL); 
        system(execute);
        end = time(NULL);
        total_exec_time += difftime(end, start);
    }
    exec_time = total_exec_time / niter;
    printf("Average execution time: %g seconds\n", exec_time);
    
    return 0;
}