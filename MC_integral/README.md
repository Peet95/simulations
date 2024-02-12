This program is designed to calculate the following integral using Monte Carlo integration:
$$\int_{x1}^{x2} \int_{y1}^{y2} \int_{z1}^{z2} e^{-(x^2+y^2+z^2)} dx dy dz.$$
The calculation can be performed on a CPU (single-threaded or multi-threaded) or on a GPU (multi-threaded) using the CUDA toolkit. The code was compiled with the following command: **'nvcc -O3 -use_fast_math MC.cu'**.

The limits of integration can be easily adjusted within the code, the default range is from -4.0 to 4.0 for all three variables. With these default settings, the expected result is 5.56833.

Measurements have been conducted using an Intel(R) Xeon(R) CPU @ 2.30GHz and an NVIDIA Tesla T4 16GB graphics card. The x-axis represents the number of data points used for the Monte Carlo integration, while the y-axis represents the corresponding running times.

![Alt text](MC_measurement.png?raw=true "Running times")

![Alt text](MC_measurement_logscale.png?raw=true "Running times on log scale")
