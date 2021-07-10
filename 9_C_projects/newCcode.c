#include<stdio.h>
#include<stdlib.h>

struct Node{
    int data;
    struct Node * next; 
};

void print12(struct Node * n){
    while(n != NULL){
        printf("%d\n", n->data);
        n=n->next;
    }
    

}

void printList(struct Node * n){
    while(n != NULL)
    {
        printf("Value= %d\n",n->data); //Value stored in the linked list data field
        printf("ptr address %d\n",n);//ptr to the address of the node
        n= n->next;
    }

}

struct Node * appendList(int n)
{
        struct Node * temp=NULL, * head=NULL, * p=NULL;

    temp = (struct Node *) malloc(sizeof(struct Node));
};


int main()
{



    first-> data=21;
    first-> next= sec;

    sec->data=3;
    sec->next=end;

    end->data=5;
    end->next=NULL;
    
    printList(first);
    print12(first);
return 0;

}
