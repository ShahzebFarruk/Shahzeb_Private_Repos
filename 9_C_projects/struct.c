#include <stdio.h>
struct Node * createLinkedList(int n);

int main(){
    struct Node
    {
        char name[100];
        struct Node* next;
        /* data */
    };
    


    return 0;
}
struct Node * createLinkedList(int n){
    int i=0;
    struct Node * head=NULL;
    struct Node * temp=NULL;
    struct Node * p=NULL;
    for (int i = 0; i < n; i++)
    {
        temp = (struct Node *) malloc(sizeof(struct Node));
        printf("enter the data\n");
        scanf("%d", &(temp->data));
        temp->next=NULL;
    

    }
    
    }