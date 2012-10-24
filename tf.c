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

double gaussian(int flag, double x)
{
  switch(flag){
    default:
    case EV:
      return exp((-1)*pow(x,2));
    case EVD:
      return (-2.0)*x*gaussian(EV, x);
  }
}

double rational_sigmoid(int flag, double x)
{
  double sxp = sqrt(1.0 + x*x);
  switch(flag){
    default:
    case EV:
      return x / (1.0 + sxp);
    case EVD:
      return 1.0 / ( sxp * (sxp + 1.0));
  }
}

