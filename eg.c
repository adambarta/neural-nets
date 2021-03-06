#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <strings.h>
#include <math.h>

#include "nn.h"
#include "cl.h"

static struct cl_trans clo;

#define KERNELDIR   "./"
#define KERNELFILE  "kernels.cl"

int main(int argc, char *argv[])
{
#if 0
  if (setup_ocl(KERNELDIR KERNELFILE, &(clo.ctx), &(clo.cq), &(clo.p)) != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "setup_ocl error\n");
#endif
    destroy(&(clo.k), &(clo.ctx), &(clo.cq), &(clo.p));
    return 1;
  }

  clo.k = get_kernel("neural_net", &(clo.p));
  if (clo.p == NULL){
#ifdef DEBUG
    fprintf(stderr, "get kernel error\n");
#endif
    destroy(&(clo.k), &(clo.ctx), &(clo.cq), &(clo.p));
    
  }
  
  
  destroy(&(clo.k), &(clo.ctx), &(clo.cq), &(clo.p));
#endif

#if 1
  struct neural_net *n;

  int layer_sizes[] = {5, 3, 2, 1, 6};
  double (*tf[])(int flag, double x) = { NULL, &gaussian, &rational_sigmoid, &sigmoid, &linear};
  
  double input[]   = {1.0, 0.5, 1.0, 1.5, 4.5};
  double desired[] = {6.0, 5.0, 4.0, 2.0, 10.0, 0.0};
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

  for (i=0; i<100; i++){
      
    error  = train_network(n, input, layer_sizes[0], desired, 0.15, 0.1);
    output = run_network(n, input, layer_sizes[0]);

    fprintf(stderr, "%d input ", i);
    for (j=0; j<layer_sizes[0]; j++){
      fprintf(stderr, "%f ", input[j]);
    }
    fprintf(stderr, "output ");
    for (j=0; j<layer_sizes[4]; j++){
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

#endif

  return 0;
}
