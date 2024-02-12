This program is to calculate the following integral using Monte Carlo integration:
$$\int_{x1}^{x2} \int_{y1}^{y2} \int_{z1}^{z2} e^{-(x^2+y^2+z^2)} dx dy dz.$$
The calculation can be made on CPU - one thread, CPU - multi thread or GPU - multi thread. The implementation of the GPU - multi thread case is made by using the CUDA toolkit. I compiled the code with the following command: nvcc -O3 -use_fast_math MC.cu.

The limits of integration can be easily changed in the code, the default is from -4.0 to 4.0 for all of the three variables. With the default setting the result should be: 5.56833.

Measurements has been made on the different solutions, with an Intel(R) Xeon(R) CPU @ 2.30GHz CPU and NVIDIA Tesla T4 16GB graphics card. The number data points used for the MC integration are on the x axis, and the running times on the y.

![Alt text](MC_measurement.png?raw=true "Running times")

![Alt text](MC_measurement_logscale.png?raw=true "Running times on log scale")
