#include <stdio.h>
#include <stdlib.h>

int main() {
    //1D arrays and single pointers
    int a[3] = {1, 2, 3};
    int *p = a;
    printf("a[0] = %d, *p = %d\n", a[0], *p);
    printf("*a = %d, p[0] = %d\n", *a, p[0]);
    //2D arrays and double pointers
    //mat
    int mat[3][3]  = {
        {1,2,3},
        {4,5,6},
        {7,8,9}
    };
    int *pmat = &mat[0][0];
    // Access elements using pointer arithmetic
    printf("Mat: \n");
    for (int i = 0; i < 3 * 3; i++) {
        printf("%d ", *(pmat + i));
    }
    printf("\n");
    //mat2
    int rows = 3;
    int cols = 3;
    int pos = 0;
    int **mat2 = calloc(rows, sizeof(int*));
    for (int row=0; row < rows; row++) {
        mat2[row] = calloc(cols, sizeof(int));
        for (int col=0; col < cols; col++) {
            mat2[row][col] = pos;
            pos++;
        }
    }
    int **pmat2 = mat2;
     // Access and print matrix elements using pointer arithmetic
    printf("Mat2 with pointer arithmetic: \n");
    //int *ip = &mat2[0][0];
    int **ip = mat2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int value = *(*(ip + i) + j);
            printf("%i ", value);
        }
        printf("\n");
    }

    printf("Mat2 with 2D array indexing: \n");
    //Access and print matrix elements with 2D array indexing
    for (int row=0; row < rows; row++) {
        printf("%d:\t", row);
        for (int col=0; col < cols; col++) {
            printf("%d ", pmat2[row][col]);
        }
        printf("\n");
    }
    //cleanup
    for (int row=0; row < rows; row++){
        free(mat2[row]);
    }
    free(mat2);
    return 0;

}