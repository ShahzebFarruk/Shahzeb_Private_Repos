
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

int main() 
{
    int a, b,s;
    a=11;
    b=0;
    b=12;
    s=a%2;
    printf("%d",s);
    if (a%2){

        printf("odd");
    }
    else if (!(a%2)){
        printf("even\n");
    }

    return 0;
}