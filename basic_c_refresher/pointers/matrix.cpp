#include <stdio.h>

int main() {
    int array1[5] = {1, 2, 3, 4, 5};
    int array2[5] = {6, 7, 8, 9, 10};
    int *pointerArray[2];

    pointerArray[0] = array1;
    pointerArray[1] = array2;

    // Print the arrays using the pointer array
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            /* NOTE: This permanently changes the pointerArray[i] value so another loop 
            over the same will not print the values*/
            printf("%d ", *pointerArray[i]++);
        }
        printf("\n");
    }

    return 0;
}