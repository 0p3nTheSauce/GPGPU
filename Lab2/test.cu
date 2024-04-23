#include <stdio.h>
#include "device_launch_parameters.h"

dim3 calculateGridSize(int nx, int ny, int blockX, int blockY)
{
    dim3 grid( ((nx + blockX - 1) / blockX) * ((ny + blockY - 1) / blockY) );
    //dim3 grid( (nx * ny) / (blockX * blockY));
    //float Fgrid = (nx * ny) / (blockX * blockY);
    // int Igrid = (nx * ny) / (blockX * blockY);
    // printf("Igrid: %d\n", Igrid);

    float Fgridx = ((nx + blockX - 1) / blockX);
    float Fgridy = ((ny + blockY - 1) / blockY);
    float Fgrid = Fgridx * Fgridy;
    printf("Fgrid: %f\n", Fgrid);
    int Igrid = ((nx + blockX - 1) / blockX) * ((ny + blockY - 1) / blockY);
    printf("Igrid: %d\n", Igrid);
    if (Fgrid > Igrid)
    {
        dim3 grid = (Igrid + 1);
    }


    return grid; 
}

int main()
{
    int nx = 1 << 12;  // 14
    int ny = 1 << 12;
    int blockX = 24;
    int blockY = 24;
    dim3 grid = calculateGridSize(nx, ny, blockX, blockY);
    printf("grid: (%d,%d)\n", grid.x, grid.y);
    return 0;
}