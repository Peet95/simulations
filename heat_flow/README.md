This C++ program is to  to solve the time-dependent heat conduction equation for a rectangular flat plate with the following boundary conditions:
  - the temperature is given at the bottom, the temperature profile is linear along the edge
  - the heat flow on the right is constant 
  - the thermal insulation along the other two edges is perfect.
The initial condition is linear in both Cartesian coordinates along the sheet, which also satisfies the boundary condition along the lower edge.

# The heat conduction equation
The observable fact in nature that heat flows from warmer areas to cooler ones can be expressed with the following mathematical equation:
```math
\boldsymbol{H} = - \kappa \nabla T(\boldsymbol{r}, t),
```
where $\boldsymbol{H}$ is the heat flux, $\kappa$ is the thermal conductivity coefficient. The total amount of heat Q(t) in the material at a given time is proportional to the integral of the temperature over the entire substance:
```math
Q(t) =  \int d\boldsymbol{r} C \rho(\boldsymbol{r}) T(\boldsymbol{r}, t),
```
where C is the specific heat capacity of the material, and $\rho$ is its density.
Since energy is a conserved quantity, the decrease in Q over time must be equal to the heat flowing out of the material. By applying energy balance and Gauss's theorem, we can derive the heat conduction equation:
```math
\frac{\partial T(\boldsymbol{r},t)}{\partial t} = \frac{\kappa}{C \rho} \nabla^2 T(\boldsymbol{r},t).
```
I have been dealing with its two-dimensional form:
```math
\frac{\partial T(x,y,t)}{\partial t} = \frac{\kappa}{C \rho} \left( \frac{\partial^2 T(x,y,t)}{\partial x^2} + \frac{\partial^2 T(x,y,t)}{\partial y^2} \right).
```
# Explicit Euler method

I conducted the simulation using the finite element method, so I discretized both space and time. In the equations, I indexed the examined time points with k, and the coordinates of the points in space with i and j indices for the x and y directions, respectively. The distance between spatial points is denoted by $\Delta x$ and $\Delta y$, and the time step is $\Delta t$.
To obtain the recursion formula used in the simulation, let's first write the first-order Taylor series expansion of $T_{i,j}^{n-1}$ around n:
```math
T_{i,j}^{n-1} = T_{i,j}^{n} - \Delta t \frac{\partial T}{\partial t} \Big\vert_{i,j}^n,
```
thus
```math
\frac{\partial T}{\partial t} \Big\vert_{i,j}^n = \frac{T_{i,j}^{n} - T_{i,j}^{n-1}}{\Delta t}.
```
In the next step, we write the second-order Taylor series expansion of $T_{i,j}^{n-1}$ and $T_{i+1,j}^{n-1}$ $T_{i-1,j}^{n-1}$:
```math
T_{i+1,j}^{n-1} = T_{i,j}^{n-1} + \Delta x \frac{\partial T}{\partial x} \Big\vert_{i,j}^{n-1} + \frac{\Delta x^2}{2!} \frac{\partial ^2 T}{\partial x^2} \Big\vert_{i,j}^{n-1}
```
```math
T_{i-1,j}^{n-1} = T_{i,j}^{n-1} - \Delta x \frac{\partial T}{\partial x} \Big\vert_{i,j}^{n-1} + \frac{\Delta x^2}{2!} \frac{\partial ^2 T}{\partial x^2} \Big\vert_{i,j}^{n-1}.
```
By adding equations the last two equations, and expressing the second derivative:
```math
 \frac{\partial ^2 T}{\partial x^2} \Big\vert_{i,j}^{n-1} = \frac{T_{i+1,j}^{n-1} + T_{i-1,j}^{n-1} - 2 T_{i,j}^{n-1}  }{\Delta x^2}.
```
Let $\Delta x = \Delta y = h$ and substitute the obtained derivatives back into the heat conduction equation:
```math
\frac{T_{i,j}^{n} - T_{i,j}^{n-1}}{\Delta t} = \alpha \left( \frac{T_{i+1,j}^{n-1} + T_{i-1,j}^{n-1} - 4 T_{i,j}^{n-1} + T_{i,j+1}^{n-1} + T_{i,j-1}^{n-1}  }{h^2} \right).
```
Expressing the single term taken at n:
```math
T_{i,j}^{n} = T_{i,j}^{n-1} + \alpha \frac{\Delta t}{h^2} (T_{i+1,j}^{n-1} + T_{i-1,j}^{n-1} - 4 T_{i,j}^{n-1} + T_{i,j+1}^{n-1} + T_{i,j-1}^{n-1} ),
```
where $\alpha = \frac{\kappa}{C \rho}$.
