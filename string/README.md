It is a C++ program to model the vibration of a string fixed at both ends. I visualized the data output of the program with the Gnuplot visualization tool.

# The motion equation of the vibrating string
The equation of motion to be solved is the one-dimensional wave equation:
```math
\frac{\partial y(x,t)}{\partial x^2} = \frac{1}{c^2} \frac{\partial y(x,t)}{\partial t^2}.
```
If we also want to consider damping proportional to velocity, it modifies to the following:
```math
\frac{\partial ^2 y(x,t)}{\partial x^2} = \frac{1}{c^2} \frac{\partial ^2 y(x,t)}{\partial t^2} - A \frac{\partial  y(x,t)}{\partial t}.
```
To truly model a string fixed at both ends with a wave equation, boundary conditions need to be considered. For simplicity, let the string be of unit length and fixed at both ends, then:
```math
u(0, t) = 0 \;\;\;\;\;\;\; u(1, t) = 0.
```

In addition, to obtain a specific particular solution, we also need to determine initial conditions. In general, we could determine the shape of the string at time 0 and its time derivative, but for simplicity, we set the time derivative throughout to 0, so:
```math
u(x, 0) = f(x) \;\;\;\;\;\;\; \frac{\partial u}{\partial t}u(x, 0) = 0.
```

We can numerically solve the equation by discretizing the problem. At any given time, we calculate the solution from the values obtained at the previous two time steps, i.e., we apply recursion. The second derivatives can be defined as follows:

```math
\frac{\partial ^2 y(t,x)}{dx^2} = \frac{ y(t, x+dx) - 2 y(t,x) + y(t,   x-dx) }{dx^2}
```
```math
\frac{\partial ^2 y(t,x)}{dt^2} = \frac{ y(t+dt,x   ) - 2 y(t,x) + y(t-dt,x   ) } {dt^2 }.
```

Substituting these back into the wave equation:
```math
 y(t+dt,x) = ( c^2  dt^2 / dx^2 )  y(t,  x+dx)  +  2  ( 1 - C^2  dt^2 / dx^2 )  y(t,   x   )  +\\+  (     c^2  dt^2 / dx^2 )  y(t,   x-dx)   -  y(t-dt,x   ).
```
```math
 y(t+dt,x) = ( c^2  dt^2 / dx^2 )  y(t,  x+dx)  +  2  ( 1 - C^2  dt^2 / dx^2 )  y(t,   x   )  +\\+ (     c^2  dt^2 / dx^2 )  y(t,   x-dx)   -  y(t-dt,x   ) - \frac{ A}{dt} ( y(t,   x   ) - y(t-dt,x   )).
```
However, we cannot compute the value after the first time step in this way because it requires two previous results, but we only have the initial condition. For the first step, we can use the central difference formula:

```math
\frac{dy}{dt(t,x)} = \frac{ y(t+dt,x) - y(t-dt,x) } {  2 dt }.
```

So we can approximate $y(t-dt,x)$ as follows:
```math
y(t-dt,x) = y(t+dt,x) - 2  dt  \frac{ dy}{dt(t,x)}
```

Replacing the $y(t-dt,x)$ value in the equation:
```math
  y(t+d,x) =   1/2  (C^2  dt^2 / dx^2 )  y(t,   x+dx) +   ( 1 - C^2  dt^2 / dx^2 )  y(t,   x   ) +\\+ 1/2  (     C^2  dt^2 / dx^2 )  y(t,   x-dx) +  dt \frac{dy}{dt(t,   x   )}.
```
The first time step can thus be performed in this way.

# Analytical solution in the form of a Fourier series
The solution can be sought by separating the variables:
```math
y(x,t) = X(x)T(t)
```
Substituting this into the equation, we get a harmonic oscillator motion equation for both the spatial and temporal terms:
```math
X(x) = A \sin{k x} + B \cos{k x}
```
```math
T(t) = C \sin{\omega t} + D \cos{\omega t}
```
Taking into account that at the initial time, the displacement must be zero at the origin, the B coefficient in the X(x) function must be zero. Also, considering that the other end of the string is also fixed, the A coefficient and the wave number k can only take certain values. In the T(t) function, the C coefficient must be zero, due to the second initial condition:
```math
X(x) = A_n \sin{k_n x} \;\;\;\;\;\;\; k_n = \pi n
```
```math
T(t) = D_n \cos{\omega_n t} \;\;\;\;\;\;\; \omega_n = c k_n = c \pi n.
```

Substituting these into the equation, we get:
```math
y_n (x,t) =  A_n \sin{k_n x} D_n \cos{\omega_n t} = B_n \sin{k_n x} \cos{\omega_n t}.
```

Since the equation is linear in y, the general solution can be written as the sum of normal modes:
```math
y (x,t) = \sum_{n=1}^{\infty} B_n \sin{k_n x} \cos{\omega_n t}.
```
The $B_n$ coefficients can be determined from the first initial condition:

```math
y (x,0) = \sum_{n=1}^{\infty} B_n \sin{k_n x} = f(x)
```



# Simulation with different initial conditions, with and without damping

Using the time stepping mentioned before, I calculated the displacement of the string at various locations at different time points.
I took the wave propagation velocity to be $c = 0.25$ everywhere, and the distance between the two fixed ends of the string to be one unit length.

First, I ran the simulation with the following initial condition:
```math
    y(x,0)=
\begin{cases}
      1.25 x & ,\text{if}\ x \leq 0.8 \\
      5-5x & ,\text{if}\ x > 0.8,
      \end{cases}
```
The results obtained this way:

![1_new](https://github.com/Peet95/projects/assets/128177702/9b7fb453-24ab-4567-82eb-2c8917ba660c)

With damping:

![1_damped_new](https://github.com/Peet95/projects/assets/128177702/2037ea61-8c12-400b-83f3-5ba47598aea0)




It's worth examining what the solution will be for the following initial conditions:
```math
   y(x,0)=
   \begin{cases}
      ( x - 0.25 )  ( 0.5 - x ) & ,\text{if}\ 0.25 \leq x \leq 0.5 \\
      ( x - 0.75 )  ( 1 - x ) & ,\text{if}\ 0.75 \leq x \leq 1 \\
      0 & ,\text{otherwise}.
    \end{cases}
```
 
The result of this can be seen here:

![2_new](https://github.com/Peet95/projects/assets/128177702/34c41990-7fa7-4ba8-b5cf-c7b8445d2c46)

With damping:

![2_damped_new](https://github.com/Peet95/projects/assets/128177702/d02b076b-169b-4fd4-8da6-066d555ea2ca)


# Simulation with different initial conditions, numerically and analytically

By substituting the (\ref{eq:iv1}) initial condition into the (\ref{sth}) equation, we obtain the following result for the $B_n$ coefficients:

```math
B_n = 2 \int\limits_0^{0.8} 1.25 x \sin{(n \pi x)} dx + 2 \int\limits_{0.8}^1 (5 - 5x) \sin{(n \pi x)} dx = \frac{12.5 \sin{(0.8 n \pi)} - 10  \sin {(n \pi)}} {n^2 \pi^2}.
```

With this, we can calculate the analytical solution and compare it with the previously obtained numerical one, as shown here:

![3_new](https://github.com/Peet95/projects/assets/128177702/87ad6e22-bcbf-4f2f-b3b0-43b57da296ed)

Here the green is the numerical result and purple is the analtical one.
To make the two solutions more comparable, it is worth looking at the x-z plane at certain time points, as shown on this picture:

![4](https://github.com/Peet95/projects/assets/128177702/8d7950b9-2416-4b11-9a0c-54aeecacae0e)

![5](https://github.com/Peet95/projects/assets/128177702/0d50b5f3-c018-45a8-8a4c-4beedfa6f5ea)

It can be seen that the two match quite well.

