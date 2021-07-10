#include<stdio.h>
int sumOf(int A[]){

}
int main(){
    int A[5]={1,21,3,5,7,'0'};

    int * p;
    p=A;
    int a=0;
    a=sizeof(A)/sizeof(A[0]);
    printf("r= %d\n",a);
    for(int i=0; i<=a;i++)
    {

        printf("A[%d]=%d, Addr &A[i]=%d or A+i = %d \n",i,*(A+i),&A[i], (A+i));
    }
    
printf("%d",*(A+4));


    return 0;

}