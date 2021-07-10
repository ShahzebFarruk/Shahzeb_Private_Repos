#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


int main() 
{
    int a, b;
    //printf("fsdfsdf");
    scanf("%d\n%d", &a, &b);
      // Complete the code.
      char number [9][7] = {"one","two","three","four","five","six","seven","eight","nine"};
      char o[20][34]= {"odd","even"};
      int label_index;
      for( int i=a; i<=b; i++){
          if(i<=9){
              label_index=i-1;
            printf("%s\n",number[label_index]);
          }
          else if (i%2){
            printf("odd\n");}
          else if (!(i%2)){
            printf("even\n");
                 
      }}




    return 0;
}
