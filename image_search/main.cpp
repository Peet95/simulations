// g++ main.cpp

#include<iostream>
#include<fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

float calcCorr(uint8_t* X, uint8_t* Y, int X_width, int X_height, int Y_width, int width_offset, int height_offset) {
    float sum_X = 0, sum_Y = 0, sum_XY = 0;
    float squareSum_X = 0, squareSum_Y = 0;

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
    float corr = (sum_XY * X_width*X_height - (sum_X * sum_Y))
                  / std::sqrt((squareSum_X * X_width*X_height - (sum_X * sum_X))
                      * (squareSum_Y * X_width*X_height - (sum_Y * sum_Y)));
    //std::cout << "corr: " << corr << std::endl;
    return corr;
}

int main() {
    int width, width_small, height, height_small, bpp;

    uint8_t* rgb_image = stbi_load("office.png", &width, &height, &bpp, 3);
    uint8_t* rgb_image_small = stbi_load("owl.png", &width_small, &height_small, &bpp, 3);

    int N = 3*width*height;
    int sz = N * sizeof(float);

    float corr{0.0};
    int x_place{0};
    int y_place{0};
    for (int width_offset = 0; width_offset < (width - width_small)*3; width_offset++) {
        for (int height_offset = 0; height_offset < (height - height_small); height_offset++) {
            float corr_temp = calcCorr(rgb_image_small, rgb_image, width_small*3, height_small, width*3, width_offset, height_offset);
            if (corr_temp > corr) {
                corr = corr_temp;
                x_place = width_offset;
                y_place = height_offset;
            }
        }
    }
    std::cout << "\ncorr: " << corr << std::endl;
    std::cout << "x_place: " << x_place << std::endl;
    std::cout << "y_place: " << y_place << std::endl;

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