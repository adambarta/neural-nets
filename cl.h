#ifndef SHARED_H
#define SHARED_H

#include <CL/opencl.h>

#if 0
struct float2 {
  float x;
  float y;
};

typedef struct float2 float2;
#endif

#define CL_SUCCESS 0

struct cl_trans {
  cl_context ctx;
  cl_command_queue cq;
  cl_program p;
  cl_kernel k;
};

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

const char* oclErrorString(cl_int error);

int setup_ocl(char *kf, cl_context *context, cl_command_queue *command_queue, cl_program *program);

void destroy(cl_kernel *kernel, cl_context *context, cl_command_queue *command_queue, cl_program *program);

cl_kernel get_kernel(char *name, cl_program *p);

#endif

