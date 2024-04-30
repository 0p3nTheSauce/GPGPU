/* Program to compute Pi using Monte Carlo methods */
/* Compile with gcc mcpi.c -o mcp */
/* Run with mcp */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define SEED 35791246

int main(int argc, char **argv)
{
   int niter=0;
   double x,y;
   int i,count=0; /* # of points in the 1st quadrant of unit circle */
   double z;
   double pi;

   if (argc != 2)
   {
      printf("Usage: MCPi <number_of_iterations>\n");
      exit(EXIT_FAILURE);
   } 
   niter = atoi(argv[1]);
   if (niter <= 0) {
      printf("Number of iterations must be a positive integer\n");
      exit(EXIT_FAILURE);
   }
   /* initialize random numbers */
   srand(SEED);
   count=0;
   for ( i=0; i<niter; i++) {
      x = (double)rand()/RAND_MAX;
      y = (double)rand()/RAND_MAX;
      z = x*x+y*y;
      if (z<=1) count++;
      }
   pi=(double)count/niter*4;
   printf("# of trials= %d , estimate of pi is %g \n",niter,pi);
   return 0;
}
