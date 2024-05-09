#include <stdio.h>
#include <stdlib.h>



// void printMatrix(int rows, int cols, int matrix[rows][cols]) {
void printMatrix(int rows, int cols, int **matrix) {
    for (int row=0; row < rows; row++) {
        printf("%d:\t", row);
        for (int col=0; col < cols; col++) {
            printf("%d ", matrix[row][col]);
        }
        printf("\n");
    }
}

void printMatrixLinear(int rows, int cols, int **matrix) {
    // for (int ix = 0, iy = 0; ix < rows, iy <  cols; ix++, iy++){
    //     int i = ix + iy * cols;
    //     printf("%d:\t", ix);
    //     printf("%d \n", *(*matrix + i));
    // }

    
}

int main(int argc, char **argv) {
    const int rows = 5;
    const int cols = 5;
    //int matrix1[rows][cols];
    //int *matrix1[rows];

    int **matrix1 = calloc(rows, sizeof(int*));
    for (int row=0; row < rows; row++) {
        matrix1[row] = calloc(cols, sizeof(int));
        for (int col=0; col < cols; col++) {
            matrix1[row][col] = row + col;
        }
    }

    // for (int row=0; row < rows; row++) {
    //     for (int col=0; col < cols; col++) {
    //         matrix1[row][col] = row + col;
    //     }
    // }
    int **pmat = matrix1;
    printf("sizeof(matrix1) = %lu\n", sizeof(matrix1));
    printMatrix(rows, cols, pmat);
    printMatrixLinear(rows, cols, pmat);

    // for (int row=0; row < rows; row++){
    //     free(matrix1[row]);
    // }
    // free(matrix1);
}


