#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int pfunc(int *x, int *y) {
    int lx = *x;
    int ly = *y;
    lx = lx + 5;
    ly = ly + 5;
    *x = lx;
    *y = ly;
    // *x = *x + 5;
    // *y = *y + 5;
    // printf("In function: \n");
    // printf("x: %d   y: %d\n", *x, *y);
}

int main() {
    //pointers and functions
    int x = 5;
    int y = 4;
    pfunc(&x, &y);
    printf("x: %d   y: %d\n", x, y);

    //1D arrays and single pointers
    int a[3] = {1, 2, 3};
    int *p = a;
    int b[3];
    memcpy(b, p, sizeof(a));
    printf("a[0] = %d, *p = %d\n", a[0], *p);
    printf("*a = %d, p[0] = %d\n", *a, p[0]);
    printf("b[0] = %d, *b = %d\n", a[0], *p);
    //2D arrays and double pointers
    //mat
    int rows = 3;
    int cols = 3;
    //int mat[rows][cols]  = {    not allowed
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
    // int rows = 3;
    // int cols = 3;
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