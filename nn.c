#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <math.h>

#include "nn.h"

double sigmoid(int flag, double x)
{
  switch(flag){
    default:
    case EV:
      return 1.0 / (1.0 + exp((-1.0)*x));
    case EVD:
      return sigmoid(EV, x)*(1.0 - sigmoid(EV, x));
  }
}

double linear(int flag, double x)
{
  switch(flag){
    default:
    case EV:
      return x;
    case EVD:
      return 1.0;
  }
}

double gaussian()
{
  static double u, v, s, t;
  static int p = 0;

  do {
    
    u = 2.0 * (double)rand() / RAND_MAX - 1.0;
    v = 2.0 * (double)rand() / RAND_MAX - 1.0;

    s = u*u + v*v;

  } while (s > 1.0 || (u == 0.0 && v == 0.0));

  t = sqrt(-2.0 * log(s) / s) * ( (p==0) ? u : v );

  p = 1 - p;

  return t;
}

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

struct neural_net *create_neural_net(int *layer_sizes, double (**tf)(int flag, double x), int l_len)
{ 
  struct neural_net *n;
  int i, j, k;

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
  
  for (i=0; i<n->layer_cnt; i++){
    
    for (j=0; j<n->layer_size[i]; j++){
      
      n->bias[i][j]            = gaussian();
      n->prev_bias_delta[i][j] = 0.0;
      n->layer_outputs[i][j]   = 0.0;
      n->layer_inputs[i][j]    = 0.0;
      n->delta[i][j]           = 0.0;

    }

    for (k=0; k < ((i == 0) ? n->input_size : n->layer_size[i-1]); k++){

      for (j=0; j<n->layer_size[i]; j++){
        n->weight[i][k][j]            = gaussian();  
        n->prev_weight_delta[i][k][j] = 0.0;  
      }

    }

  }


  return n;
}


double *run_network(struct neural_net *n, double *input, int ilen)
{
  int l, i, j;
  double sum, *output;

  if (n == NULL)
    return NULL;

  if (ilen != n->input_size){
#ifdef DEBUG
    fprintf(stderr, "e: run input len and network configuration missmatch!\n");
#endif
    return NULL;
  }

  output = malloc(sizeof(double) * n->layer_size[n->layer_cnt-1]);
  if (output == NULL){
    return NULL;
  }
  
  bzero(output, sizeof(double) * n->layer_size[n->layer_cnt-1]);
  
  for (l=0; l<n->layer_cnt; l++){

    for (j=0; j<n->layer_size[l]; j++){
      
      sum = 0.0;

      for (i=0; i< (l==0? n->input_size : n->layer_size[l-1]); i++){
        sum += n->weight[l][i][j] * (l==0 ? input[i] : n->layer_outputs[l-1][i]);
      }

      sum += n->bias[l][j];

      n->layer_inputs[l][j]  = sum;
      if (n->trans_func[l])
        n->layer_outputs[l][j] = (*(n->trans_func[l]))(EV, sum);
      else
        n->layer_outputs[l][j] = 0.0;

    }

  }
  
  for (i=0; i<n->layer_size[n->layer_cnt-1]; i++){
    output[i] = n->layer_outputs[n->layer_cnt-1][i];
  }

  return output;
}

double train_network(struct neural_net *n, double *input, int ilen, double *desired, double trate, double momentum)
{
  double error, sum, weight_delta, bias_delta;
  double *output;

  int l,i,j,k;

  if (n == NULL || input == NULL || desired == NULL || n->input_size != ilen)
    return 0.0;

  error         = 0.0;
  sum           = 0.0;
  weight_delta  = 0.0;
  bias_delta    = 0.0;

  output = run_network(n, input, ilen);
  if (output == NULL)
    return 0.0;
  
  for (l = n->layer_cnt - 1; l >= 0; l--){
    
    if (l == n->layer_cnt - 1){
      //output layer
      for (k=0; k<n->layer_size[l]; k++){
        n->delta[l][k] = output[k] - desired[k];
        error += pow(n->delta[l][k], 2);
        if (n->trans_func[l])
          n->delta[l][k] *= (*(n->trans_func[l]))(EVD, n->layer_inputs[l][k]);
        else 
          n->delta[l][k] *= 0.0;
      }

    } else { 
      //hidden layer
      for (i=0; i<n->layer_size[l]; i++){
        sum = 0.0;
        for(j=0; j< n->layer_size[l+1]; j++){
          sum += n->weight[l+1][i][j] * n->delta[l+1][j];
        }

        if (n->trans_func[l])
          sum *= (*(n->trans_func[l]))(EVD, n->layer_inputs[l][i]);
        else 
          sum *= 0.0;

        n->delta[l][i] = sum;
      }
    }
 
  }
  
  for (l = 0; l < n->layer_cnt; l++)
    for(i = 0; i<(l==0?n->input_size:n->layer_size[l-1]); i++)
      for (j=0; j<n->layer_size[l]; j++){
        weight_delta = trate * n->delta[l][j] * (l==0?input[i]:n->layer_outputs[l-1][i])
                      + momentum * n->prev_weight_delta[l][i][j];
        n->weight[l][i][j] -= weight_delta;
#if 0
        n->weight[l][i][j] -= weight_delta + momentum * n->prev_weight_delta[l][i][j];
#endif
        n->prev_weight_delta[l][i][j] = weight_delta ;
      }

  for (l = 0; l < n->layer_cnt; l++)
    for (i = 0; i<n->layer_size[l]; i++){
      bias_delta  = trate * n->delta[l][i];
      n->bias[l][i] -= bias_delta + momentum * n->prev_bias_delta[l][i];
      n->prev_bias_delta[l][i] = bias_delta;
    }

  if (output)
    free(output);

  return error;  
}



int main(int argc, char *argv[])
{
  struct neural_net *n;

  int layer_sizes[] = {1, 3, 1};
  double (*tf[])(int flag, double x) = { NULL, &sigmoid, &linear};
  
  double input[]   = {1.0};
  double desired[] = {2.5};
  double *output;
  double error;

  int ilen;
  ilen  = sizeof(layer_sizes) / sizeof(int);
  
  error = 0.0;

  n = create_neural_net(layer_sizes, tf, ilen);
  if (n == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: create neural network failed\n");
#endif
    return 1;
  }

  int i, j;

  for (i=0; i<10000; i++){
      
    error = train_network(n, input, layer_sizes[0], desired, 0.15, 0.1);
    output = run_network(n, input, layer_sizes[0]);
    fprintf(stderr, "%d input ", i);
    for (j=0; j<layer_sizes[0]; j++){
      fprintf(stderr, "%f ", input[j]);
    }
    fprintf(stderr, "output ");
    for (j=0; j<layer_sizes[2]; j++){
      fprintf(stderr, "%f ", output[j]);
    }
    fprintf(stderr, "error %f\n", error);
    

    free(output);
    
  }
  


  destroy_neural_net(n);

#if 0
  int i;
  for (i=0; i<10000; i++){
    fprintf(stdout, "%f\n", gaussian());
  }
#endif

  return 0;
}


