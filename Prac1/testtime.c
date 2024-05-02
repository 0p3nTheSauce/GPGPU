#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    int sleep_seconds = atoi(argv[1]);
    printf("Calling the sleep function for %d seconds\n", sleep_seconds);
    printf("Timestamp: %d\n", (int)time(NULL));
    sleep(sleep_seconds);
    printf("Timestamp: %d\n", (int)time(NULL));
    return 0;
}