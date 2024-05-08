#include <stdio.h>

// Function prototype
void printMatrix(int *matrix, int rows, int cols);

int main() {
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Call the function to print the matrix
    printMatrix(*matrix, 3, 3);

    return 0;
}

// Function definition to print the matrix
void printMatrix(int *matrix, int rows, int cols) {
    printf("Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(matrix + i * cols + j));
        }
        printf("\n");
    }
}
