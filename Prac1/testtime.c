#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    // time_t start, end;
    int sleep_seconds = atoi(argv[1]);
    printf("Calling the sleep function for %d seconds\n", sleep_seconds);
    // start = time(NULL);
    sleep(sleep_seconds);
    // end = time(NULL);
    // printf("Time taken to print sum is %.2f seconds\n",
    //        difftime(end, start));

    return 0;
}