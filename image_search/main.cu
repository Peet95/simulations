// nvcc main.cu -diag-suppress 550

#include<iostream>
#include<fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void calcCorr(float* result, uint8_t* X, uint8_t* Y, int X_width, int X_height, int Y_width)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    float sum_X = 0, sum_Y = 0, sum_XY = 0;
    float squareSum_X = 0, squareSum_Y = 0;

    int width_offset = id % Y_width;
    int height_offset = id / Y_width;

    for (int i = 0; i < X_height; i++) {
        for (int j = 0; j < X_width; j++) {
            // sum of elements of array X.
            sum_X = sum_X + X[X_width * i + j];

            // sum of elements of array Y.
            sum_Y = sum_Y + Y[Y_width * (i + height_offset) + j + width_offset];

            // sum of X[i] * Y[i].
            sum_XY = sum_XY + X[X_width * i + j] * Y[Y_width * (i + height_offset) + j + width_offset];

            // sum of square of array elements.
            squareSum_X = squareSum_X + X[X_width * i + j] * X[X_width * i + j];
            squareSum_Y = squareSum_Y + Y[Y_width * (i + height_offset) + j + width_offset] * Y[Y_width * (i + height_offset) + j + width_offset];
        }
    }

    // use formula for calculating correlation coefficient.
    result[id] = (sum_XY * X_width*X_height - (sum_X * sum_Y))
                  / std::sqrt((squareSum_X * X_width*X_height - (sum_X * sum_X))
                      * (squareSum_Y * X_width*X_height - (sum_Y * sum_Y)));
                      
}

int main() {
    int width, width_small, height, height_small, bpp;
    int x_place{0}, y_place{0};

    uint8_t* rgb_image = stbi_load("office.png", &width, &height, &bpp, 3);
    uint8_t* rgb_image_small = stbi_load("owl.png", &width_small, &height_small, &bpp, 3);

    int N = 3*width*height;
    int sz = N * sizeof(float);
  
    // Host vector
    float* h_dst = (float*)malloc(sz);
    uint8_t* h_X = (uint8_t*)malloc(3*width_small*height_small*sizeof(uint8_t));
    uint8_t* h_Y = (uint8_t*)malloc(3*width*height*sizeof(uint8_t));
    // Device vector
    float* d_dst = NULL;
    uint8_t* d_X = NULL;
    uint8_t* d_Y = NULL;    
    // Allocate memory for the vector on GPU
    cudaMalloc((void**)&d_dst, sz);
    cudaMalloc((void**)&d_X, 3*width_small*height_small*sizeof(uint8_t));
    cudaMalloc((void**)&d_Y, 3*width*height*sizeof(uint8_t));

    // Initialize vector on host
    for (int i = 0; i < N; i++) {
        h_dst[i] = 0.0;
        h_Y[i] = rgb_image[i];
    }

    for (int i = 0; i < 3*width_small*height_small; i++) {
        h_X[i] = rgb_image_small[i];
    }

    // Copy host vector to device
    cudaMemcpy(d_dst, h_dst, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, 3*width_small*height_small*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, 3*width*height*sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Number of thread blocks in grid
    int threadsPerBlock = 512;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    calcCorr << <blocksPerGrid, threadsPerBlock >> > (d_dst, d_X, d_Y, width_small*3, height_small, width*3);

    // Copy array back to host
    cudaMemcpy(h_dst, d_dst, sz, cudaMemcpyDeviceToHost);

    float corr{0.0};
    for (int i =0 ; i < N; i++) {
      float corr_temp = h_dst[i];
      if(corr_temp > corr) {
        corr = corr_temp;
        x_place = i % (3*width);
        y_place = i / (3*width);
      } 
  }
    std::cout << "\ncorr: " << corr << std::endl;
    std::cout << "x_place: " << x_place / 3 << std::endl;
    std::cout << "y_place: " << y_place << std::endl;

    cudaFree(d_dst);
    free(h_dst);

    // Draw square on around the target
    for (int width_offset = 0; width_offset < 3*width_small; width_offset+=3) {
        for (int height_offset = 0; height_offset < height_small + 3; height_offset++) {
            // Top line
            rgb_image[y_place * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[y_place * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[y_place * 3*width + x_place + width_offset + 2] = 0;

            rgb_image[(y_place - 1) * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[(y_place - 1) * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[(y_place - 1) * 3*width + x_place + width_offset + 2] = 0;

            rgb_image[(y_place + 1) * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[(y_place + 1) * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[(y_place + 1) * 3*width + x_place + width_offset + 2] = 0;     

            // Bottom line
            rgb_image[(y_place + height_small) * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[(y_place + height_small) * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[(y_place + height_small) * 3*width + x_place + width_offset + 2] = 0;

            rgb_image[((y_place + height_small) - 1) * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[((y_place + height_small) - 1) * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[((y_place + height_small) - 1) * 3*width + x_place + width_offset + 2] = 0;

            rgb_image[((y_place + height_small) + 1) * 3*width + x_place + width_offset + 0] = 255;
            rgb_image[((y_place + height_small) + 1) * 3*width + x_place + width_offset + 1] = 0;
            rgb_image[((y_place + height_small) + 1) * 3*width + x_place + width_offset + 2] = 0;  

            // Left line
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place - 3] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place - 2] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place - 1] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 1] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 2] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 4] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 5] = 0;            

            // Right line
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small - 3] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small - 2] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small - 1] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small + 1] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small + 2] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small + 3] = 255;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small + 4] = 0;
            rgb_image[(y_place + height_offset - 1) * 3*width + x_place + 3*width_small + 5] = 0;                                             
        }
    }

    stbi_write_png("office_out.png" , width, height, 3, rgb_image, width*3);

    stbi_image_free(rgb_image);
    stbi_image_free(rgb_image_small);

    return 0;
}