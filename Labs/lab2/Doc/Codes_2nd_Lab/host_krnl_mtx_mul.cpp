#include "xcl2.hpp"
#include "event_timer.hpp"
#include <algorithm>
#include <vector>

//OUR CODE//
#define lm 4
#define ln 4
#define lp 4

#define m  (1 << lm)
#define n  (1 << ln)
#define p  (1 << lp)


int main(int argc, char **argv) {
int A[n][m];
int B[m][p];
int C_HW[n][p];
int C_SW[n][p];

// Compute the size of array in bytes // το είχε και στο vadd
    size_t size_in_bytes_a = m * n * sizeof(int);
    size_t size_in_bytes_b = m * p * sizeof(int);
    size_t size_in_bytes_res = p * n * sizeof(int);
  // όπως το είχε και αυτος
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  EventTimer et;
  std::string binaryFile = argv[1];
  cl_int err;
  cl::Context context;

// This call will get the kernel object from program. A kernel is an
// OpenCL function that is executed on the FPGA.
  cl::Kernel krnl_mtx_mul; // το αλλαζεις /// οριζεις ονομα του kernel
  cl::CommandQueue q; // μαλλον ιδιο 

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


  // μεταφορα δεδομενων απο τον host για να τα εχει ο kernel ετσι ωστε να μπορει να τρεξει τη συναρτηση, dram στην μνημη του fpga και απο κει στις Bram
  // sw κανονικη c

  et.add("Allocate Memory in Host Memory");
  std::vector<int, aligned_allocator<int>> source_in1(n*m); // τα αλλαζεις 
  std::vector<int, aligned_allocator<int>> source_in2(m*p);
  std::vector<int, aligned_allocator<int>> source_hw_results(n*p);
  et.finish(); // όπως ηταν

// Create the test data
  et.add("Fill the buffers"); // οπως ηταν
// αρχικοποιω τα 
// Set Matrix A
  for(int i = 0; i < n; i++){
     for(int j = 0; j < m; j++){
     	source_in1[(i*m)+j] = rand() % 256; // δισδιαστατο σε vector
     	A[i][j] = source_in1[(i*m)+j];
     }
  }
// Set Matrix B
  for(int i = 0; i < m; i++){
      for(int j = 0; j < p; j++){
    	  source_in2[(i*p)+j] = rand() % 256;
    	  B[i][j] = source_in2[(i*m)+j];
      }
  }
//Set C_SW and C_HW with zeros
  for(int i = 0; i < n; i++){
	  for(int j = 0; j < p; j++){
		  source_hw_results[(i*p) +j] = 0;
		  C_HW[i][j] = 0;
		  C_SW[i][j] = 0;
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
      ////////////////
      OCL_CHECK(err, krnl_mtx_mul = cl::Kernel(program, "MATRIX_MUL_HW", &err));
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
					 size_in_bytes_a, source_in1.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_in2(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
					 size_in_bytes_b, source_in2.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
					 size_in_bytes_res, source_hw_results.data(), &err));
  et.finish();

  // SET ARGUMENTS OF KERNEL
  et.add("Set the Kernel Arguments");
  int size = DATA_SIZE;
  OCL_CHECK(err, err = krnl_mtx_mul.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_mtx_mul.setArg(1, buffer_in2));
  OCL_CHECK(err, err = krnl_mtx_mul.setArg(2, buffer_output));
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
  OCL_CHECK(err, err = q.enqueueTask(krnl_mtx_mul));
  et.finish();

  // Copy Result from Device Global Memory to Host Local Memory
  et.add("Copy Result from Device Global Memory to Host Local Memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
  OCL_CHECK(err, err = q.finish());
  et.finish();
  // OPENCL HOST CODE AREA END

  //MATRIX_MUL_SW(A, B, C_SW);
  for(int i = 0; i < n; i++){
          for(int j = 0; j < p; j++){
               int res = 0;
               for(int k = 0; k < m; k++){
                  res += A[i][k] * B[k][j];
               }
               C_SW[i][j] = res;
          }
   }

  // Compare the results of the Device to the simulation
  et.add("Compare the results of the Device to the simulation");
  bool match = true;
  for (int i = 0; i < n; i++) {
	  for(int j =0; j<m ; j++){
		  if (source_hw_results[i*m + j] != C_SW[i][j]) {
			  std::cout << "Error: Result mismatch" << std::endl;
			  std::cout << "i = " << i << " CPU result = " << C_SW[i][j]
                << " Device result = " << source_hw_results[i] << std::endl;
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
