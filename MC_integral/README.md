This program is to calculate the following integral using Monte Carlo integration:
$$\int_{x1}^{x2} \int_{y1}^{y2} \int_{z1}^{z2} e^{-(x*x+y*y+z*z)} dx dy dz$$.

The calculation can be made on CPU - one thread, CPU - multi thread or GPU - multi thread.

The implementation of the GPU - multi thread case is made by using the CUDA toolkit.

The limits of integration can be easily changed in the code, the default is from -4.0 to 4.0 for all of the three variables. With the default setting the result should be: 5.56833.

I made some time measurements on the different solutions. The CPU was an Intel(R) Xeon(R) CPU @ 2.30GHz and graphic card was an NVIDIA Tesla T4 16GB.

![Alt text](MC_measurement.png?raw=true "Title")
