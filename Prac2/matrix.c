#include <stdio.h>
#include <stdlib.h>



// void printMatrix(int rows, int cols, int matrix[rows][cols]) {
void printMatrixHeap(int rows, int cols, int **matrix) {
    for (int row=0; row < rows; row++) {
        printf("%d:\t", row);
        for (int col=0; col < cols; col++) {
            printf("%d ", matrix[row][col]);
        }
        printf("\n");
    }
}

void printMatrixStack(int rows, int cols, int *matrix) {
    int c = 0;
    for (int i = 0; i < rows * cols; i++) {
        printf("%d ", *(matrix + i));
        c++;
        if (c == cols){
            printf("\n");
        }
    }
}

int  *init(int rows, int cols) {
    int v = 0;
    int matrix[rows][cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = v;
            v++;
        }
    }
    int *p =&matrix[rows][cols];
}

int main(int argc, char **argv) {
    const int rows = 5;
    const int cols = 4;
    //Heap allocated
    int **matrix1 = calloc(rows, sizeof(int*));
    int i = 0;
    for (int row=0; row < rows; row++) {
        matrix1[row] = calloc(cols, sizeof(int));
        for (int col=0; col < cols; col++) {
            matrix1[row][col] = i;
            i++;
        }
    }
    int **pmat = matrix1;
    printf("sizeof(matrix1) = %lu\n", sizeof(matrix1));
    printMatrixHeap(rows, cols, pmat);
    //Stack allocated 
    // int *matrix2 = init(rows, cols);
    int matrix2[rows][cols] = {
        {0, 1, 2, 3}, 
        {4, 5, 6, 7}, 
        {8, 9, 10, 11}, 
        {12, 13, 14, 15},
        {16, 17, 18, 19} 
    };
    int *p = &matrix2[0][0];
    printMatrixStack(rows, cols, p);
    //clean up heap
    for (int row=0; row < rows; row++){
        free(matrix1[row]);
    }
    free(matrix1);

}


