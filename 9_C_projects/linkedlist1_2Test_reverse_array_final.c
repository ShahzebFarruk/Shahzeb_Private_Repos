#include<stdio.h>
struct node
{
    int data;
    struct node * next;
};

struct node * createlinkedlist(int * array, int count){

    struct node * head=NULL, *temp=NULL,*temp1=NULL, *p=NULL;
    //temp=(struct node *) malloc(sizeof(struct node));
    //temp->data= * array;
   //temp->next=NULL;
    
        // Reverse a linked list using arrays
        while(count--!=0){
            //printf("count= %d\n ", count);
            temp1=(struct node *) malloc(sizeof(struct node));
            temp1->data= * (array + count);
            temp1->next=NULL;
            printf("temp1 -> data %d\n", temp1->data);
            //printf("temp1 %d\n", temp1);
            if(head==NULL){
                    head=temp1;
            }
            else{
            p=head;
            printf("p= %d\n", p);
            while(p->next!=NULL){
                p=p->next;
            }
            p->next=temp1;
            
            }
            //count=count-1;
    }
return head;
}

void printList(struct node * n){
    while(n != NULL)
    {
        printf("Value= %d\n",n->data); //Value stored in the linked list data field
        printf("ptr address %d\n",n);//ptr to the address of the node
        n= n->next;
    }

}
int main(){
    int array[2];
    printf("enter the number, count= %d\n", sizeof(array)/sizeof(array[0]));
    for (int i = 0; i < sizeof(array)/sizeof(array[0]); i++)
    {
    scanf("%d",&array[i]);
    
    }
    int count;
    count=sizeof(array)/sizeof(array[0]);   
    struct node * head=NULL;
    head=createlinkedlist(&array, count);
    printList(head);
    
    
    return 0;
}