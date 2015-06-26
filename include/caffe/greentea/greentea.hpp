/*
 * greentea.hpp
 *
 *  Created on: Apr 5, 2015
 *      Author: Fabian Tschopp
 */

#ifndef CAFFE_GREENTEA_HPP_
#define CAFFE_GREENTEA_HPP_

// Define ViennaCL/GreenTea flags
#ifdef USE_GREENTEA
#ifndef NDEBUG
#define NDEBUG
#endif

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif

#include "CL/cl.h"
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"
#endif

#ifndef GREENTEA_QUEUE_COUNT
#define GREENTEA_QUEUE_COUNT 1

#endif

namespace caffe {

#ifdef USE_GREENTEA
viennacl::ocl::handle<cl_mem> WrapHandle(cl_mem in,
                                         viennacl::ocl::context *ctx);
#endif

enum Backend {
  BACKEND_CUDA,
  BACKEND_OpenCL
};

class DeviceContext {
 public:
  DeviceContext();
  DeviceContext(int id, Backend backend);
  Backend backend() const;
  int id() const;
 private:
  int id_;
  Backend backend_;
};

void FinishQueues(viennacl::ocl::context *ctx);

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

#ifdef USE_GREENTEA

#ifdef USE_VIENNACLBLAS
#define GREENTEA_VCL_BLAS_CHECK(condition) \
    {ViennaCLStatus status = condition; \
    CHECK_EQ(status, ViennaCLSuccess) << "GreenTea ViennaCL BLAS ERROR";}
#endif

#ifdef USE_CLBLAS
#define GREENTEA_CL_BLAS_CHECK(condition) \
    {clblasStatus status = condition; \
    CHECK_EQ(status, clblasSuccess) << "GreenTea CL BLAS ERROR";}
#endif

// Macro to select the single (_float) or double (_double) precision kernel
#define CL_KERNEL_SELECT(kernel) \
  is_same<Dtype, float>::value ? \
      kernel "_float" : \
      kernel "_double"

#endif

}  // namespace caffe

#endif  /* CAFFE_GREENTEA_HPP_ */