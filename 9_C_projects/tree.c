#include<stdio.h>

struct tree
{
  int root;
  struct tree * left;
  struct tree * right;
};

struct node* insert(struct node *root, int x){
    //new node created
    struct tree * temp= (struct tree* ) malloc(sizeof(struct tree));
    temp->root=x;
    temp->left=NULL;
    temp->right=NULL;
    if(root==NULL){
        return insert(x);
    }

}
void display(struct tree * head){

    while(head!= NULL){
        printf("%d  ", &(head->root));
    }
}

int main(){
    struct tree * head= NULL;
    insert(5);
    insert(6);
    insert(7);
    display(head);

    return 0;

}