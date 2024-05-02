#include <stdio.h>
#include <time.h>
int main(void)
{
    time_t start, end;
    start = time(NULL);
    int a, b;
    scanf("%d %d", &a, &b);
    printf("Sum of %d and %d is %d\n",
           a, b, a + b);
    end = time(NULL);
    printf("Time taken to print sum is %.2f seconds",
           difftime(end, start));
}