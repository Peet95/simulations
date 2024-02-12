%%writefile MC.cu
/* This program is for Monte Carlo integration of
   exp(-(x*x+y*y+z*z)) dx dy dz  x=x1..x2 y=y1..y2 z=z1..z2
   and to compare the calculation times in case of
   CPU or GPU multi thread. */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <curand_kernel.h>
#include <ctime>

__host__ __device__ float r2(const float& x, const float& y, const float& z) {
     return (x * x + y * y + z * z);
}

float MonteCarlo_SingleThread(const int& N, const float& x1, const float& x2, const float& y1, const float& y2, const float& z1, const float& z2) {
    float px, py, pz;
    float sum{0.0};
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution_x(x1, x2);
    std::uniform_real_distribution<float> distribution_y(y1, y2);
    std::uniform_real_distribution<float> distribution_z(z1, z2);

    for (int i = 0; i < N; i++) {
        px = distribution_x(gen);
        py = distribution_y(gen);
        pz = distribution_z(gen);
        if (r2(px, py, pz) < 16.0) {
		       sum += exp(-r2(px, py, pz));
	      }
    }
    float integral = (x2 - x1) * (y2 - y1) * (z2 - z1) * sum / N;
    return integral;
}

auto summation = [](const int &N, const float &x1, const float &x2, const float &y1, const float &y2, const float &z1, const float &z2) {
    float px, py, pz;
    float sum {0.0};
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution_x(x1, x2);
    std::uniform_real_distribution<float> distribution_y(y1, y2);
    std::uniform_real_distribution<float> distribution_z(z1, z2);
    for (int i = 0; i < N; i++) {
        px = distribution_x(gen);
        py = distribution_y(gen);
        pz = distribution_z(gen);
        if (r2(px, py, pz) < 16.0) {
		       sum += exp(-r2(px, py, pz));
	      }
    }
    return sum;
};

float MonteCarlo_CPUMultiThread(const int& N, const float &x1, const float &x2, const float &y1, const float &y2, const float &z1, const float &z2) {
    int max_num_of_threads = (int) std::thread::hardware_concurrency();
    int N_thread = N / max_num_of_threads;
    std::vector<std::future<float>> futures(max_num_of_threads);

    for (int i = 0; i < max_num_of_threads; ++i) {
        futures[i] = std::async(std::launch::async, summation, std::ref(N_thread), std::ref(x1), std::ref(x2), std::ref(y1), std::ref(y2), std::ref(z1), std::ref(z2));
    }
    auto sum = std::accumulate(futures.begin(), futures.end(), 0.0, [](float acc, std::future<float> &f) {
        return acc + f.get();
    });

    float integral = (x2 - x1) * (y2 - y1) * (z2 - z1) * sum / N;
    return integral;
}

__global__ void calculate(float* dst, const float &x1, const float &x2, const float &y1, const float &y2, const float &z1, const float &z2)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;

    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);

    float px = curand_uniform(&state) * (x2 - x1) + x1;
    float py = curand_uniform(&state) * (y2 - y1) + y1;
    float pz = curand_uniform(&state) * (z2 - z1) + z1;

    if (r2(px, py, pz) < 16.0) {
      dst[id] = (exp(-r2(px, py, pz)));
    }
}

void ZeroInitialization(const int& N, float* A) {
    for (int i = 0; i < N; i++) {
        A[i] = 0.0;
    }
}

void RandomInitialization(const int& N, float* A) {
    std::random_device rd{};
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(-4.0, 4.0);
    for (int i = 0; i < N; i++) {
        A[i] = distribution(gen);
    }
}

float MonteCarlo_GPUMultiThread(const int& N, const float &x1, const float &x2, const float &y1, const float &y2, const float &z1, const float &z2) {
    // Size, in bytes, of each vector
    size_t sz = N * sizeof(float);
    auto time2 = std::chrono::high_resolution_clock::now();

    // Host vector
    float* h_dst = (float*)malloc(sz);

    // Device vector
    float* d_dst = NULL;

    // Allocate memory for the vector on GPU
    cudaMalloc((void**)&d_dst, sz);

    // Initialize vector on host
    for (int i = 0; i < N; i++) {
        h_dst[i] = 0.0;
    }

    // Copy host vector to device
    cudaMemcpy(d_dst, h_dst, sz, cudaMemcpyHostToDevice);

    // Number of thread blocks in grid
    int threadsPerBlock = 512;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    calculate << <blocksPerGrid, threadsPerBlock >> > (d_dst, x1, x2, y1, y2, z1, z2);

    // Copy array back to host
    cudaMemcpy(h_dst, d_dst, sz, cudaMemcpyDeviceToHost);

    // Calculate sum
    float sum = 0.0;
    for(int i = 0; i < N; i++) {
      sum += h_dst[i];
    }
    float integral = 8.0 * 8.0 * 8.0 * sum / N;

    cudaFree(d_dst);
    free(h_dst);

    return integral;
}

int main() {
    int choose, N;
    float result;

    do {
      std::cout << "Choose calculation type:\n0 - CPU Single Thread\n1 - CPU Multi Thread\n2 - GPU Multi Thread\n3 - Time analysis" << std::endl;
      std::cin >> choose;
    } while (choose < 0 || choose > 3);

    if(choose == 3) {
        std::ofstream file;
        file.open("result.txt");
        file << std::setw(20) << std::left << "#N" << std::setw(20) << std::left << "CPUsingle";
        file << std::setw(20) << std::left << "CPUmulti" << std::setw(20) << std::left << "GPUmulti" << std::endl;
        for (N = 600; N < 600'000'001; N *= 10) {
            std::cout << "N = " << N << std::endl;
            auto time0 = std::chrono::high_resolution_clock::now();
            MonteCarlo_SingleThread(N, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            auto time1 = std::chrono::high_resolution_clock::now();

            auto time2 = std::chrono::high_resolution_clock::now();
            MonteCarlo_CPUMultiThread(N, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            auto time3 = std::chrono::high_resolution_clock::now();

            auto time4 = std::chrono::high_resolution_clock::now();
            MonteCarlo_GPUMultiThread(N, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            auto time5 = std::chrono::high_resolution_clock::now();

            file << std::setw(20) << std::left << N << std::setw(20) << std::left << std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count();
            file << std::setw(20) << std::left << std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
            file << std::setw(20) << std::left << std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() << std::endl;
        }
        file.close();
        std::cout << "Results saved into 'result.txt'" << std::endl;
        return 0;
    }

    std::cout << "Number of points:" << std::endl;
    std::cin >> N;

    auto time0 = std::chrono::high_resolution_clock::now();

    switch (choose) {
        case 0:
            result = MonteCarlo_SingleThread(N, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            break;
        case 1:
            result = MonteCarlo_CPUMultiThread(N, -4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            break;
        case 2:
            result = MonteCarlo_GPUMultiThread(N,-4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
            break;
    }

    auto time1 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() << " microseconds" << std::endl;
    std::cout << "Result: " << result;
  
    return 0;
}
