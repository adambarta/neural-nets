#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "nn.h"

void destroy_neural_net(struct neural_net *n)
{
  if (n){
      
    
      
    
    free(n);
  }
}

struct neural_net *create_neural_net(int *layer_sizes, int l_len, double *(*tf)(double x), int t_len)
{ 
  struct neural_net *n;
  int i;

  if (layer_sizes == NULL || tf == NULL)
    return NULL;

  n = malloc(sizeof(struct neural_net));
  if (n == NULL)
    return NULL;

  n->input_size       = layer_sizes[0];
  n->layer_cnt        = l_len - 1;
  
  /*1d*/
  n->layer_size = malloc(sizeof(int) * n->layer_cnt);
  if (n->layer_size == NULL){
    destroy_neural_net(n);
    return NULL;
  }
  for (i=0; i < n->layer_cnt; i++){
    n->layer_size[i] = layer_sizes[i+1];
  }


  n->trans_func = malloc(sizeof(double *) * n->layer_cnt);
  if (n->trans_func == NULL){
    destroy_neural_net(n);
    return NULL;
  }
  for (i=0; i < n->layer_cnt; i++){
    n->trans_func[i] = tf[i+1];
  }


  /*2d*/
  n->bias = malloc(sizeof(double *) * n->layer_cnt);
  if (n->bias == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  n->prev_bias_delta  = malloc(sizeof(double *) * n->layer_cnt);
  if (n->prev_bias_delta == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  n->delta = malloc(sizeof(double *) * n->layer_cnt);
  if (n->delta == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  n->layer_outputs = malloc(sizeof(double *) * n->layer_cnt);
  if (n->layer_outputs == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  n->layer_inputs = malloc(sizeof(double *) * n->layer_cnt);
  if (n->layer_inputs == NULL){
    destroy_neural_net(n);
    return NULL;
  }


  /*3d*/
  n->weight           = NULL;
  n->prev_weight_delta= NULL;

  return n;
}




int main(int argc, char *argv[])
{


  
  return 0;
}


