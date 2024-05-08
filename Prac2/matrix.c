#include <stdio.h>

#define ROWS 3
#define COLS 3


// Function prototype
void printMatrix(int matrix[][COLS]);

int main() {
    // Declare a 2D array (matrix) with ROWS rows and COLS columns
    int matrix[ROWS][COLS];

    // Initialize the matrix (optional)
    // For example, let's initialize it with consecutive numbers
    int count = 1;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i][j] = count++;
        }
    }
    printMatrix(matrix);
    

    return 0;
}

// Function definition to print the matrix
void printMatrix(int matrix[][COLS]) {
    printf("Matrix:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}