#include<stdio.h>
#include <stdlib.h>
struct node
{
    int data;
    struct node * next;

};


void printList(struct node * n){
    while(n != NULL)
    {
        printf("Value= %d\n",n->data); //Value stored in the linked list data field
        printf("ptr address %d\n",n);//ptr to the address of the node
        n= n->next;
    }

}
struct node * addendLinkedList(int n){
    int i=0;
    struct node * head=NULL, * temp=NULL, *p=NULL;
    
    
    for (i=0;i<n;i++)
    {
        temp =(struct node*) malloc(sizeof(struct node));
        scanf("%d",&(temp->data));
        temp->next=NULL;

    if (head == NULL){
        head=temp;
    }
    else {
        p= head;
        while(p->next!=NULL){
            p=p->next;
        }
        p->next=temp;
    }
    
    }    
    
return head;


}

int main(){
    struct node * head= NULL;
    int n=0;
    
    scanf("%d", &n);
    head=addendLinkedList(n);
    printf("%d",head);
    printList(head);
    printf("%d",head);
    



    return 0;
}