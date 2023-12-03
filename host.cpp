
#include "xcl2.hpp"
#include "event_timer.hpp"
#include <algorithm>
#include <vector>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <cstdint>




#define LM 4
#define LN 4
#define LP 4

#define M ( 1 << LM )
#define N ( 1 << LN )
#define P ( 1 << LP )


int main(int argc, char **argv) {
int a[M][N], b[N][P];
int result_sw[M][P];

size_t size_a = M * N * sizeof(int);
size_t size_b = N * P * sizeof(int);
size_t size_result_hw = M * P * sizeof(int);





  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  EventTimer et;

  std::string binaryFile = argv[1];

  cl_int err;
  cl::Context context;
  cl::Kernel krnl_vector_add;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR
  et.add("Allocate Memory in Host Memory");
  std::vector<int, aligned_allocator<int>> source_a(size_a);
  std::vector<int, aligned_allocator<int>> source_b(size_b);
  std::vector<int, aligned_allocator<int>> source_result_hw(size_result_hw);
  et.finish();

  // Create the test data
  et.add("Fill the buffers");
  // inits
  for(int i = 0 ; i < M ; i++){
      for(int j = 0 ; j < N ; j++){
    	 source_a[(i*N)+j] = (int)(rand() % 256);
         a[i][j] = source_a[(i*N)+j];
      }
  }

  for(int i = 0 ; i < N ; i++){
      for(int j = 0 ; j < P ; j++){
    	 source_b[(i*P)+j] = (int)(rand() % 256);
         b[i][j] = source_b[(i*P)+j];
      }
  }

  for(int i = 0 ; i < M ; i++){
       for(int j = 0 ; j < P ; j++){
    	 source_result_hw[(i*P)+j] = 0;
         result_sw[i][j] = 0;
       }
   }

  et.finish();

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  et.add("Load Binary File to Alveo U200");
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vadd", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }
  et.finish();

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  et.add("Allocate Buffer in Global Memory");
  OCL_CHECK(err, cl::Buffer buffer_in1(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     size_a, source_a.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_in2(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     size_b, source_b.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
					 size_result_hw, source_result_hw.data(), &err));
  et.finish();

  et.add("Set the Kernel Arguments");

  OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_in2));
  OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_output));
  et.finish();

  // Copy input data to device global memory
  et.add("Copy input data to device global memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));
  et.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel
  et.add("Launch the Kernel");
  OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
  et.finish();

  // Copy Result from Device Global Memory to Host Local Memory
  et.add("Copy Result from Device Global Memory to Host Local Memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
  OCL_CHECK(err, err = q.finish());
  et.finish();
  // OPENCL HOST CODE AREA END

	 for(int i = 0 ; i < M ; i++){
	        for(int j = 0; j < P ; j++){
	           result_sw[i][j] = 0;
	           for (int k = 0; k < N; ++k)
	                result_sw[i][j] += a[i][k] * b[k][j];
	        }
	    }

  // Compare the results of the Device to the simulation
  et.add("Compare the results of the Device to the simulation");
  bool match = true;
  for (int i = 0; i < M; i++) {
	  for(int j = 0; j < P ; j++)
	  {
		  if (source_result_hw[(i*P + j)] != result_sw[i][j]) {
		       std::cout << "Error: Result mismatch" << std::endl;
		       std::cout << "i = " << i << " CPU result = " << result_sw[i][j]
		                 << " Device result = " << source_result_hw[i] << std::endl;
		       match = false;
		       break;
	  }
    }
  }

  et.finish();

  std::cout <<"----------------- Key execution times -----------------" << std::endl;
  et.print();

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}






