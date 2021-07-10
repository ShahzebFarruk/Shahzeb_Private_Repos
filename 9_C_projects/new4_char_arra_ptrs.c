#include<stdio.h>
void p(int c[]){
    printf("%d\n",c);
}

int main(){
    char c1[20]="Shahzeb";
    //printf("%c",c1);
    p(c1);


    return 0;
}