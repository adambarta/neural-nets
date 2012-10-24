#ifndef NN_H
#define HH_H

#define EV  0 /*evaluate*/
#define EVD 1 /*evaluate derivative*/


struct neural_net {
  
  int input_size;
  int layer_cnt;
  int *layer_size;

  double (**trans_func)(int flag, double x);

  double **layer_outputs;
  double **layer_inputs;
  double **bias;
  double **delta;
  double **prev_bias_delta;

  double ***weight;
  double ***prev_weight_delta;
  
};


double sigmoid(int flag, double x);
double linear(int flag, double x);
double gaussian(int flag, double x);
double rational_sigmoid(int flag, double x);


struct neural_net *create_neural_net(int *layer_sizes, double (**tf)(int flag, double x), int l_len);

void destroy_neural_net(struct neural_net *n);

double train_network(struct neural_net *n, double *input, int ilen, double *desired, double trate, double momentum);

double *run_network(struct neural_net *n, double *input, int ilen);



#endif
