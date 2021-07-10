#include <stdio.h>

int main(){
    int B[2][3]={{1,2,3},{2,3,4}};
    //int z[2][3];
    printf("Main hit\n\n");


    for(int i=0; i<2;i++)
    {
    for(int j=0;j<3;j++)
    {
        printf("%d\t", *(B+i)+j);
        
    }
    printf("\n");}
    printf("%d", *B);

    return 0;

}