#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <math.h>

#include "nn.h"

void destroy_neural_net(struct neural_net *n)
{
  int i,k;

  if (n){

    if (n->trans_func){
      free(n->trans_func);
    }
    
    for (i=0; i<n->layer_cnt; i++){

      if (n->layer_outputs && n->layer_outputs[i])
        free(n->layer_outputs[i]);

      if (n->layer_inputs && n->layer_inputs[i])
        free(n->layer_inputs[i]);

      if (n->bias && n->bias[i])
        free(n->bias[i]);
      
      if (n->delta && n->delta[i])
        free(n->delta[i]);

      if (n->prev_bias_delta && n->prev_bias_delta[i])
        free(n->prev_bias_delta[i]);
      
      for(k=0; k < ((i == 0) ? n->input_size : n->layer_size[i-1]); k++){
        
        if (n->weight && n->weight[i] && n->weight[i][k])
          free(n->weight[i][k]);

        if (n->prev_weight_delta && n->prev_weight_delta[i] && n->prev_weight_delta[i][k])
          free(n->prev_weight_delta[i][k]);
        
      }  

      if (n->weight && n->weight[i])
        free(n->weight[i]);
          
      if (n->prev_weight_delta && n->prev_weight_delta[i])
        free(n->prev_weight_delta[i]);

    }

    if (n->weight)
        free(n->weight);
          
    if (n->prev_weight_delta)
        free(n->prev_weight_delta);

    if (n->layer_outputs)
      free(n->layer_outputs);

    if (n->layer_inputs)
      free(n->layer_inputs);

    if (n->bias)
      free(n->bias);

    if (n->delta)
      free(n->delta);

    if (n->prev_bias_delta)
      free(n->prev_bias_delta);

    if (n->layer_size)
      free(n->layer_size);
    
    free(n);
  }
}

struct neural_net *create_neural_net(int *layer_sizes, double (**tf)(double x), int l_len)
{ 
  struct neural_net *n;
  int i, k;

  if (layer_sizes == NULL || tf == NULL)
    return NULL;

#ifdef DEBUG
  fprintf(stderr, "%s: layers [%d]\n", __func__, l_len);
#endif

  n = malloc(sizeof(struct neural_net));
  if (n == NULL)
    return NULL;

  bzero(n, sizeof(struct neural_net));



  n->input_size       = layer_sizes[0];
  n->layer_cnt        = l_len - 1;
  n->layer_size       = NULL;
  n->trans_func       = NULL;
  n->bias             = NULL;
  n->prev_bias_delta  = NULL;
  n->delta            = NULL;
  n->layer_outputs    = NULL;
  n->layer_inputs     = NULL;
  n->weight           = NULL;
  n->prev_weight_delta= NULL;



  /*1d*/
  n->layer_size = malloc(sizeof(int) * n->layer_cnt);
  if (n->layer_size == NULL){
    destroy_neural_net(n);
    return NULL;
  }
  n->trans_func = malloc(sizeof(double *) * n->layer_cnt);
  if (n->trans_func == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  for (i=0; i < n->layer_cnt; i++){
    n->layer_size[i] = layer_sizes[i+1];
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
  n->weight = malloc(sizeof(double **) * n->layer_cnt);
  if (n->weight == NULL){
    destroy_neural_net(n);
    return NULL;
  }

  n->prev_weight_delta = malloc(sizeof(double **) * n->layer_cnt);
  if (n->prev_weight_delta == NULL){
    destroy_neural_net(n);
    return NULL;
  }




  for (i=0; i<n->layer_cnt; i++){
    n->bias[i] = malloc(sizeof(double) * n->layer_size[i]);
    if(n->bias[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }

    n->prev_bias_delta[i] = malloc(sizeof(double) * n->layer_size[i]);
    if(n->prev_bias_delta[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }

    n->delta[i] = malloc(sizeof(double) * n->layer_size[i]);
    if(n->delta[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }
    
    n->layer_outputs[i] = malloc(sizeof(double) * n->layer_size[i]);
    if(n->layer_outputs[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }

    n->layer_inputs[i] = malloc(sizeof(double) * n->layer_size[i]);
    if(n->layer_inputs[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }

    n->weight[i] = malloc(sizeof(double *) * ((i == 0)?n->input_size:n->layer_size[i-1]));
    if (n->weight[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }
    
    n->prev_weight_delta[i] = malloc(sizeof(double *) * ((i == 0)?n->input_size:n->layer_size[i-1]));
    if (n->prev_weight_delta[i] == NULL){
      destroy_neural_net(n);
      return NULL;
    }

    for(k=0; k < ((i == 0) ? n->input_size : n->layer_size[i-1]); k++){
      n->weight[i][k] = malloc(sizeof(double) * n->layer_size[i]);
      if(n->weight[i][k] == NULL){
        destroy_neural_net(n);
        return NULL;
      }

      n->prev_weight_delta[i][k] = malloc(sizeof(double) * n->layer_size[i]);
      if(n->prev_weight_delta[i][k] == NULL){
        destroy_neural_net(n);
        return NULL;
      }

    }
     
  }
  
  return n;
}


double sigmoid(double x)
{
  return 1.0 / (1.0 + exp((-1.0)*x));
}


int main(int argc, char *argv[])
{
  struct neural_net *n;

  int input[] = {2, 3, 2, 1};
  double (*tf[])(double x) = {&sigmoid, &sigmoid, &sigmoid, &sigmoid};
  
  int ilen;

  ilen  = sizeof(input) / sizeof(int);

  n = create_neural_net(input, tf, ilen);
  if (n == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: create neural network failed\n");
#endif
    return 1;
  }

  destroy_neural_net(n);

  return 0;
}


