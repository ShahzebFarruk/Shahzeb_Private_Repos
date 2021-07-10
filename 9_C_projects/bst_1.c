#include<stdio.h>
struct bst_node
{
    int data;
    struct bst_node * left, * right;
};

struct bst_node * createnode(int data){
    struct bst_node * temp=NULL;
    temp=(struct node *) malloc(sizeof(struct bst_node));
    temp->data=data;
    temp->left=NULL;
    temp->right=NULL;
return temp;
}

struct bst_node* insert(struct bst_node *root, int x)
{
    //searching for the place to insert
    if(root==NULL)
        return createnode(x);
    else if(x>root->data) // x is greater. Should be inserted to right
        root->right = insert(root->right, x);
    else // x is smaller should be inserted to left
        root->left = insert(root->left,x);
    return root;
}

void print(struct bst_node * head){
    while(head->right && head->left !=NULL){
       // printf("\t\t\t\t %d",head->root);
        //printf("\n");
        //printf("\\");
       // printf("%d",head->left);
        //printf("%d",head->right);
        
    }
}

void inorder(struct bst_node *root)
{
    if(root!=NULL) // checking if the root is not null
    {
        inorder(root->left); // visiting left child
        printf(" %d ", root->data); // printing data at root
        inorder(root->right);// visiting right child
    }
}

int main(){
    struct bst_node * head=NULL, * root=NULL;
    head=createnode(10);
root = createnode(20);
    insert(root,5);
    insert(root,1);
    insert(root,15);
    insert(root,9);
    insert(root,7);
    insert(root,12);
    insert(root,30);
    insert(root,25);
    insert(root,40);
    insert(root, 45);
    insert(root, 42);
        //createnode(head, data);
    inorder(root);
    printf("\n");
    
    return 0;

}