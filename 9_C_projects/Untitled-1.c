#include <stdio.h>
#include <stdlib.h>

void m(int * p){
    *p=100;
    printf("m p= %d\n",p);
    printf("%d\n",&p);
    printf("%d\n",*p);


}

int main(){
    //printf("fsdf\n");

    struct Node
    {
        int data;
        struct Node* link;
        /* data */
    };
    

    int z=144;
    int* z_p;
    z_p=&z;
    m(&z);

    printf("\nz= %d\n",z);
    printf("%d\n", &z); 
    printf("z_p= %d\n",z_p);
    printf("z_p* = %d\n",*z_p);

    printf("z_p& = %d\n",&z_p);
    char *p0;
    p0= (char*) z_p;  //typecasting
    printf("size of char %d", sizeof(p0+1));
    printf("\n%d", p0);
    printf("\n%d", p0+1);
    printf("\n%d", *(p0+1));



    return 0;

}
