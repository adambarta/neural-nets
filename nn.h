#ifndef NN_H
#define HH_H

struct neural_net {
  
  int input_size;
  int layer_cnt;
  int *layer_size;

  double (**trans_func)(double x);

  double **layer_outputs;
  double **layer_inputs;
  double **bias;
  double **delta;
  double **prev_bias_delta;

  double ***weight;
  double ***prev_weight_delta;
  
};

struct neural_net *create_neural_net(int *layer_sizes, double (**tf)(double x), int l_len);
void destroy_neural_net(struct neural_net *n);


#endif
