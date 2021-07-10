#include<stdio.h>
#define N 5;
int top=-1;
int stack[5];
void push(int data)
{
    if(top == 5-1){
        printf("Full");
        return;
    }
    else {
        top++;
        stack[top]=data;
    }
}


int main(){
    int a;
scanf("%d", &a);
push(a);
    return 0;
}