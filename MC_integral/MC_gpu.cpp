//apt-get install beignet (to install Intel integrated GPU driver)
//g++ MC_gpu.cpp -std=c++17 -lOpenCL -pthread -O3 -ffast-math

#include <iostream> 
#include <fstream>
#include <random>
#include <valarray>
#include <thread>
#include <future>
#include <chrono>

#include <CL/cl.hpp>

using namespace std;

int N = 100; //max.:570'000'000

void declaration2(std::valarray<float> &A){
  std::random_device rd{};
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution(-4.0, 4.0);
  for(int i = 0; i < N; i++) {
    A[i] = distribution(gen);
  }
}

void MonteCarlo_Hybrid() { 
  //get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if(all_platforms.size()==0){
    std::cout<<" No platforms found. Check OpenCL installation!\n";
    exit(1);
  }
  std::cout << "Number of platforms found: "<< all_platforms.size() << "\n";
  for (const auto& platform : all_platforms) std::cout << "Found platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
  cl::Platform default_platform=all_platforms[0];
  std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
     
  //get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size()==0){
    std::cout<<" No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  std::cout << "Number of devices found: "<< all_devices.size() << "\n";
  cl::Device default_device=all_devices[0];
  std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
       
  cl::Context context({default_device});
     
  cl::Program::Sources sources;
     
  //kernel
  std::string kernel_code=	     
                "   void kernel MC(global float* A, global const float* B, global const float* C){       "
                "     int gid = get_global_id(0);       "
                "     if(A[gid]*A[gid]+B[gid]*B[gid]+C[gid]*C[gid]<16.0){ A[gid]=exp(-A[gid]*A[gid]-B[gid]*B[gid]-C[gid]*C[gid]);}      "
                "   }                                                                               ";
  sources.push_back({kernel_code.c_str(),kernel_code.length()});
     
  cl::Program program(context,sources);
  if(program.build({default_device})!=CL_SUCCESS){
    std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
    exit(1);
  }   
  std::valarray<float> A(0.0, N);
  std::valarray<float> B(0.0, N);
  std::valarray<float> C(0.0, N);
  
  std::vector<std::future<void>> futures(3);

  futures[0] = std::async( std::launch::async, declaration2, ref(A));
  futures[1] = std::async( std::launch::async, declaration2, ref(B));
  futures[2] = std::async( std::launch::async, declaration2, ref(C));

  futures[0].get();
  futures[1].get();
  futures[2].get();

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, default_device);
     
  //write arrays A, B, C to the device
  cl::Buffer buffer_A{ context, std::begin(A), std::end(A), true };
  cl::Buffer buffer_B{ context, std::begin(B), std::end(B), true };
  cl::Buffer buffer_C{ context, std::begin(C), std::end(C), true };
     
  //run the kernel
  cl::Kernel kernel_add=cl::Kernel(program,"MC");
  kernel_add.setArg(0, buffer_A);
  kernel_add.setArg(1, buffer_B);
  kernel_add.setArg(2, buffer_C);

  queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(N),cl::NullRange);
  queue.finish();

  //read result from the device to array A
  cl::copy(queue, buffer_A, std::begin(A), std::end(A));

  float sum = std::accumulate(std::begin(A), std::end(A), 0.0); // initialize sum 
  float integral = 8.0*8.0*8.0*sum / N; 
  cout << "Result: " << integral << endl;
}

void MonteCarlo_GPUMultiThread() { 
  //get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if(all_platforms.size()==0){
    std::cout<<" No platforms found. Check OpenCL installation!\n";
    exit(1);
  }
  std::cout << "Number of platforms found: "<< all_platforms.size() << "\n";
  for (const auto& platform : all_platforms) std::cout << "Found platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
  cl::Platform default_platform=all_platforms[0];
  std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
     
  //get default device of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size()==0){
    std::cout<<" No devices found. Check OpenCL installation!\n";
    exit(1);
  }
  std::cout << "Number of devices found: "<< all_devices.size() << "\n";
  cl::Device default_device=all_devices[0];
  std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
       
  cl::Context context({default_device});
     
  cl::Program::Sources sources;
     
  // kernel
  std::string kernel_code=	     
                "   void kernel MC(global float* A, global const float* B, global const float* C){       "
                "     int gid = get_global_id(0);       "
                "     if(A[gid]*A[gid]+B[gid]*B[gid]+C[gid]*C[gid]<16.0){ A[gid]=exp(-A[gid]*A[gid]-B[gid]*B[gid]-C[gid]*C[gid]);}      "
                "   }                                                                               ";
  sources.push_back({kernel_code.c_str(),kernel_code.length()});
     
  cl::Program program(context,sources);
  if(program.build({default_device})!=CL_SUCCESS){
    std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
    exit(1);
  }   
  // create buffers on the device
  std::valarray<float> A(0.0, N);
  std::valarray<float> B(0.0, N);
  std::valarray<float> C(0.0, N);

  std::random_device rd{};
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution_x(-4.0, 4.0);
  std::uniform_real_distribution<float> distribution_y(-4.0, 4.0);
  std::uniform_real_distribution<float> distribution_z(-4.0, 4.0);

  for(int i = 0; i < N; i++) {
    A[i] = distribution_x(gen);
    B[i] = distribution_y(gen);
    C[i] = distribution_z(gen);
  }
     
  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context,default_device);
     
  //write arrays A, B, C to the device
  cl::Buffer buffer_A{ context, std::begin(A), std::end(A), true };
  cl::Buffer buffer_B{ context, std::begin(B), std::end(B), true };
  cl::Buffer buffer_C{ context, std::begin(C), std::end(C), true };
     
  //run the kernel
  cl::Kernel kernel_add=cl::Kernel(program,"MC");
  kernel_add.setArg(0,buffer_A);
  kernel_add.setArg(1,buffer_B);
  kernel_add.setArg(2,buffer_C);

  queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(N),cl::NullRange);
  queue.finish();

  //read result from the device to array A
  cl::copy(queue, buffer_A, std::begin(A), std::end(A));

  float sum = std::accumulate(std::begin(A), std::end(A), 0.0); // initialize sum
  float integral = 8.0*8.0*8.0*sum / N; 
  cout << "Result: " << integral << endl;
}

float MonteCarlo_SingleThread(auto&& lambda1, auto&& lambda2, float x1, float x2, float y1, float y2, float z1, float z2) { 
  float px, py, pz;
  float sum = 0.0;
  std::random_device rd{};
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution_x(x1, x2);
  std::uniform_real_distribution<float> distribution_y(y1, y2);
  std::uniform_real_distribution<float> distribution_z(z1, z2);
  for(int i = 0; i < N; i++) {
    px = distribution_x(gen);
    py = distribution_y(gen);
    pz = distribution_z(gen);
    if(lambda2(px, py, pz)) sum += lambda1(px, py, pz);
  }
  std::cout << sum << "\n";
  float integral = (x2-x1)*(y2-y1)*(z2-z1)*sum / N; 
  cout << "Result: " << integral << endl;
}

auto summation =  [](auto&& lambda1, auto&& lambda2, int& N, float& x1, float& x2, float& y1, float& y2, float& z1, float& z2) {  
  float px, py, pz;
  float sum = 0.0;   
  std::random_device rd{};
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution_x(x1, x2);
  std::uniform_real_distribution<float> distribution_y(y1, y2);
  std::uniform_real_distribution<float> distribution_z(z1, z2);
  for(int i = 0; i < N; i++) {
    px = distribution_x(gen);
    py = distribution_y(gen);
    pz = distribution_z(gen);
    if(lambda2(px, py, pz)) sum += lambda1(px, py, pz);
  }
  return sum;
};

void MonteCarlo_CPUMultiThread(auto&& lambda1, auto&& lambda2, float x1, float x2, float y1, float y2, float z1, float z2) {
  int max_num_of_threads = (int)std::thread::hardware_concurrency();
  std::cout << "Max. number of threads: " << max_num_of_threads << endl;
  int N_thread = N/max_num_of_threads;
  std::vector<std::future<float>> futures(3);

  for(int n=0; n<3; ++n ) {
    futures[n] = std::async( std::launch::async, summation, ref(lambda1), ref(lambda2), ref(N_thread), ref(x1), ref(x2), ref(y1), ref(y2), ref(z1), ref(z2));
  }
  auto parallel_result = std::accumulate(futures.begin(), futures.end(), 0.0, [](float acc, std::future<float>& f){ return acc + f.get(); } );

  float integral = (x2-x1)*(y2-y1)*(z2-z1)*parallel_result / N; 
  cout << "Result: " << integral << endl;
}

int main(){
   cout << "Choose calculation type:\n0 - CPU Single Thread\n1 - CPU Multi Thread\n2 - GPU Multi Thread\n3 - CPU & GPU Hybrid\n4 - Time analysis" << endl;
   float result;
   int choose;
   bool route = true;
   cin >> choose;

   if(choose == 4) {
     route = false;
     ofstream file;
     file.open ("result.txt");
     file << "#N\t\t" << "CPUsingle\t\t" << "CPUmulti\t\t" << "GPU\t\t" << "Hybrid\n";
     for(N = 10; N < 200'000'000; N*=10){
       auto time0 = std::chrono::high_resolution_clock::now();
       MonteCarlo_SingleThread([](auto x, auto y, auto z){ return exp(-x*x-y*y-z*z); },
                               [](auto x, auto y, auto z)->bool{ return x*x+y*y+z*z<16.0; },
                               -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
       auto time1 = std::chrono::high_resolution_clock::now();

       auto time2 = std::chrono::high_resolution_clock::now();
       MonteCarlo_CPUMultiThread([](auto x, auto y, auto z){ return exp(-x*x-y*y-z*z); },
                               [](auto x, auto y, auto z)->bool{ return x*x+y*y+z*z<16.0; },
                               -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
       auto time3 = std::chrono::high_resolution_clock::now();

       auto time4 = std::chrono::high_resolution_clock::now();
       MonteCarlo_GPUMultiThread();
       auto time5 = std::chrono::high_resolution_clock::now();

       auto time6 = std::chrono::high_resolution_clock::now();
       MonteCarlo_Hybrid();
       auto time7 = std::chrono::high_resolution_clock::now();
       file << N << "\t\t" << std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count() << "\t\t";
       file <<                std::chrono::duration_cast<std::chrono::microseconds>(time3-time2).count() << "\t\t";
       file <<                std::chrono::duration_cast<std::chrono::microseconds>(time5-time4).count() << "\t\t";
       file <<                std::chrono::duration_cast<std::chrono::microseconds>(time7-time6).count() << "\n";
     }
     file.close();
   }

   auto time8 = std::chrono::high_resolution_clock::now();

   if(choose == 0) MonteCarlo_SingleThread([](auto x, auto y, auto z){ return exp(-x*x-y*y-z*z); },
                                           [](auto x, auto y, auto z)->bool{ return x*x+y*y+z*z<16.0; },
                                           -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
   else if(choose == 1) MonteCarlo_CPUMultiThread([](auto x, auto y, auto z){ return exp(-x*x-y*y-z*z); },
                                                  [](auto x, auto y, auto z)->bool{ return x*x+y*y+z*z<16.0; },
                                                  -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
   else if(choose == 2) MonteCarlo_GPUMultiThread();
   else if(choose == 3) MonteCarlo_Hybrid();

   else if(choose != 4) std::cout << "Not available number!\n";

   auto time9 = std::chrono::high_resolution_clock::now();
   if(route) cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(time9-time8).count() << " milliseconds" << endl;
   

   return 0;
}
