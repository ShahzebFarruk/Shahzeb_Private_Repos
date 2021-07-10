#include<stdio.h>
//Array as Function Argument
int sumOfArray(int* A, int size){ //A can be also changed as int A[] as complier creates a ptr instead of copying the entire array in fn.
    int sum=0;
    printf("%d", A);
    for (int i=0; i<size;i++)
    {

        sum=sum + *(A + i);
        printf("%d\n",A[i]);
        printf("Address = %d\n", A+i);
        printf("sum=%d\n",sum);
        
        printf(" Value = %d\n", *(A+i));
    }
}

int main(){
    int A[]= {1,2,3,4,5};
    int size=0;
    size=sizeof(A)/sizeof(A[0]);
    int total = sumOfArray(A, size);
    printf("%d",total);

    return 0;

}