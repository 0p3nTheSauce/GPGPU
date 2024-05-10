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

// Single pointer
void printMatrixSP(int rows, int cols, int *matrix) {
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(matrix + i * cols + j));
        }
        printf("\n");
    }
}

int **initHeap(int rows, int cols) {
    int **matrix = calloc(rows, sizeof(int*));
    int i = 0;
    for (int row=0; row < rows; row++) {
        matrix[row] = calloc(cols, sizeof(int));
        for (int col=0; col < cols; col++) {
            matrix[row][col] = i;
            i++;
        }
    }
    return matrix;
}

int main(int argc, char **argv) {
    const int rows = 5;
    const int cols = 4;
    //Heap allocated
    int **matrix1 = initHeap(rows, cols);
    printf("sizeof(matrix1) = %lu\n", sizeof(matrix1));
    printMatrix(rows, cols, matrix1);

    //stack allocated
    int matrix2[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11},
    };

    printMatrixSP(3, 4, *matrix2);


    //clean up heap
    for (int row=0; row < rows; row++){
        free(matrix1[row]);
    }
    free(matrix1);

}


