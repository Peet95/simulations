The purpose of this program is to simulate double pendulum using the open source freeglut library.

It is one of the most basic examples of chaotic physical systems. Chaotic systems are those whose time evolution is greatly influenced by small changes in initial values. To solve the differential equations of the double pendulum, I used the fourth order Runge-Kutta method.

The Lagrangian of the double pendulum:
```math
L = T - V = \frac{1}{2}m_1 v_1^2 + \frac{1}{2}m_2 v_2^2 - m_1 g y_1 - m_2 g y_2 =
```
```math
 = \frac{1}{2}(m_1 + m_2)l_1^2 \dot{\phi_1}^2 + \frac{1}{2} m_2 l_2^2 \dot{\phi_2}^2 + m_2 l_1 l_2 \dot{\phi_1} \dot{\phi_2} \cos(\phi2 - \phi1) - (m_1 + m_2) g l_1 \cos(\phi_1) - m_2 g l_2 \cos(\phi_2),
```

where $l_1$ is the length of the pendulum closer to the axis of rotation, $m_1$ is its mass, and $\phi_1$ is its deviation from the vertical. I marked the data of the pendulum further from the axis of rotation with the same letters, but index '2'.

The equations of motion for one or the other mass are obtained by solving the Euler-Lagrange equation:

```math
\frac{d}{dt} \frac{\partial L}{\partial \dot{\phi}_{1,2}} = \frac{\partial L}{\partial \phi_{1,2}}.
```
From this, the system of differential equations to be solved is:
```math
\frac{\partial \phi_1}{\partial t} = \omega_1
```
```math
\frac{\partial \phi_2}{\partial t} = \omega_2
```
```math
\frac{\partial \omega_1}{\partial t} = \frac{-g(M+m_1)\sin(\phi_1)-m_2 g \sin(\phi_1 -2 \phi_2)-2 \sin(\phi_1 - \phi_2) m_2 (\omega_1^2 l_1 \cos(\phi_1-\phi_2) + \omega_2^2 l_2) }{l_1 (M+m_1-m_2 \cos(2 \phi_1 - 2 \phi_2))}
```
```math
\frac{\partial \omega_2}{\partial t} = \frac{2 \sin(\phi_1 - \phi_2) (M \phi_1^2 l_1 +g M \cos(\phi_1)+ \phi_2^2 l_2 m_2 \cos(\phi_1 - \phi_2)) }{l_2 (M+m_1-m_2 \cos(2 \phi_1 - 2 \phi_2))},
```
where $M = m_1 + m_2$.

The analytical solution of this system of differential equations is not known, so we are referred to numerical methods.

If we want to take into account the effect of the medium resistance, we can calculate damping proportional to speed using a simple model. For this, we need to supplement the Lagrange formalism used in the previous chapter with the Rayleigh dissipation function:
```math
R = \frac{1}{2} \eta (v_1 ^2 + v_2 ^2) = \frac{1}{2} \eta (2 l_1^2 \dot{\phi}_1^2 + l_2^2 \dot{\phi}_2^2 + 2 l_1 l_2 \dot{\phi}_1 \dot{\phi}_2 \cos(\phi_1-\phi_2))
```

In this case, the equations of motion are obtained from the following modified Euler-Lagrange equation:
```math
\frac{d}{dt} \frac{\partial L}{\partial \dot{\phi}_{1,2}} = \frac{\partial L}{\partial \phi_{1,2}} -  \frac{\partial R}{\partial \dot{ \phi}_{1,2}}.
```

From this, the system of differential equations to be solved is:
```math
\frac{\partial \phi_1}{\partial t} = \omega_1
```
```math
\frac{\partial \phi_2}{\partial t} = \omega_2
```
```math
\frac{\partial \omega_1}{\partial t} = \frac{-g(M+m_1)\sin(\phi_1)-m_2 g \sin(\phi_1 -2 \phi_2)-2 \sin(\phi_1 - \phi2) m_2 (\omega_1^2 l_1 \cos(\phi_1-\phi_2) + \omega_2^2 l_2) }{l_1 (M+m_1-m_2 \cos(2 \phi_1 - 2 \phi_2))} - \\ \\ - \eta (2 l_1^2 \dot{\phi}_1 - l_1 l_2 \dot{\phi}_2 \cos(\phi_1 - \phi_2))
```
```math
\frac{\partial \omega_2}{\partial t} = \frac{2 \sin(\phi_1 - \phi_2) (M \phi_1^2 l_1 +g M \cos(\phi_1)+ \phi_2^2 l_2 m_2 \cos(\phi_1 - \phi_2)) }{l_2 (M+m_1-m_2 \cos(2 \phi_1 - 2 \phi_2))} - \eta (l_2^2 \dot{\phi}_2 - l_1 l_2 \dot{\phi}_1 \cos(\phi_1 - \phi_2)).
```

A short video of the running program:

https://github.com/Peet95/projects/assets/128177702/1444fe80-5505-4148-a911-9061aebf0242
