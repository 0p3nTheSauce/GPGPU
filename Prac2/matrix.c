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

void init(int rows, int cols, int **matrix) {
    int i = 0;
    for (int row=0; row < rows; row++) {
        matrix[row] = calloc(cols, sizeof(int));
        for (int col=0; col < cols; col++) {
            matrix[row][col] = i;
            i++;
        }
    }
}

int main(int argc, char **argv) {
    const int rows = 5;
    const int cols = 4;
    //Heap allocated
    int **matrix1 = calloc(rows, sizeof(int*));
    init(rows, cols, matrix1);
    printf("sizeof(matrix1) = %lu\n", sizeof(matrix1));
    printMatrix(rows, cols, matrix1);
    //clean up heap
    for (int row=0; row < rows; row++){
        free(matrix1[row]);
    }
    free(matrix1);

}


