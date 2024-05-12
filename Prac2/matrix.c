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
    //printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(matrix + i * cols + j));
            //printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}
//Single pointer subset
void printMatrixSPSub(int rows, int cols, int *matrix,
                     int fromRow, int toRow,int fromCol, int toCol) {
    //printf("Matrix:\n");
    for (int i = fromRow; i < toRow; i++) {
        for (int j = fromCol; j < toCol; j++) {
            printf("%d ", *(matrix + i * cols + j));
            //printf("%d ", matrix[i][j]);
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

//set all values of a matrix to same values
void setTo(int *matrix, int rows, int cols, int val) {
    for (int i = 0; i < rows * cols; i++) {
        *(matrix + i) = val;
    }
}

int main(int argc, char **argv) {
    const int rows = 5;
    const int cols = 4;
    //Heap allocated
    int **matrix1 = initHeap(rows, cols);
    printf("sizeof(matrix1) = %lu\n", sizeof(matrix1));
    printf("Matrix1: \n");
    printMatrix(rows, cols, matrix1);

    //stack allocated
    int matrix2[3][4] = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11},
    };
    printf("Matrix2: \n");
    printMatrixSPSub(3, 4, *matrix2, 1, 3, 2, 4);
    printf("Matrix2[1][0]: %d\n", matrix2[1][0]);

    setTo(&matrix2[0][0], 3, 4, 1);
    //printMatrixSP(rows, cols, *matrix2);
    printMatrixSPSub(3, 4, *matrix2, 0, 3, 0, 4);

    //clean up heap
    for (int row=0; row < rows; row++){
        free(matrix1[row]);
    }
    free(matrix1);

}


