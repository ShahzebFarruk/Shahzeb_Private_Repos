#include<stdio.h>
struct node
{
    int data;
    struct node * next;
};

////PRG enters the data into middle of the linked list and uses the array as input

struct node * createlinkedlist(int * array, int count){

    struct node * head=NULL, *temp=NULL,*temp1=NULL, *p=NULL;
    //temp=(struct node *) malloc(sizeof(struct node));
    //temp->data= * array;
    //temp->next=NULL;
    //head=temp;
    int i=0;
        // Straight linked list using arrays
        while(count--!=0){
            temp1=(struct node *) malloc(sizeof(struct node));
            temp1->data= * (array + i);
            printf("temp1 -> data %d\n", temp1->data);
            temp1->next=NULL;
            if (head==NULL)
            {
                head=temp1;/* code */
            }
            else{
            p=head;
            while(p->next!=NULL){
                p=p->next;
            }
            p->next=temp1;
            }
        i++;
    }
return head;
}
struct node * insert(struct node * head, int pos, int data ){
    struct node * p=NULL, * temp=NULL;
    printf("%d", pos);
    temp=(struct node *) malloc(sizeof(struct node *));
    p=head;
    
    for(int i=0; i < pos; i++) {
    if(p->next != NULL) {
        p = p->next;
    }}
    temp->data=data;
    temp->next=p->next;
    p->next=temp;

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
    int array[3];
    printf("enter the number, array size %d\n",sizeof(array)/sizeof(array[0]) );
    for (int i = 0; i < sizeof(array)/sizeof(array[0]); i++)
    {
    scanf("%d",&array[i]);
    
    }
    int count;
    count=sizeof(array)/sizeof(array[0]);   
    struct node * head=NULL;
    head=createlinkedlist(&array, count);
    printList(head);
    int pos=0, data=0;
    scanf("%d %d", &pos, &data);
    insert(head,pos-1, data);
    printList(head);
    
    return 0;
}