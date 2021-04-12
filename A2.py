# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Assignment A2 [40 marks]
# 
# The assignment consists of 3 exercises. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **discussion** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.
# %% [markdown]
# ---
# ## Question 1: Numerical Differentiation [10 marks]
# 
# A general $N$-point finite difference approximation of the derivative $F' \left( x \right)$ of a sufficiently smooth function $F \left( x \right)$ can be written as
# 
# $$
# F' \left( x \right) \approx \frac{1}{\Delta x} \sum_{i = 1}^N \alpha_i F \left( x + \beta_i \Delta x \right),
# \qquad \qquad \qquad (1)
# $$
# 
# with step size $\Delta x > 0$, and $\alpha_i, \beta_i \in \mathbb{Q}$, with $\beta_i \neq \beta_j$ for $i\neq j$. For example, the centred difference approximation $D_C(x)$ seen in the course has $N = 2$, and
# 
# $$
# \begin{cases}
# \alpha_1 = \frac{1}{2}, &\alpha_2 = -\frac{1}{2}, \\
# \beta_1 = 1, &\beta_2 = -1,
# \end{cases}
# \qquad
# \text{giving} \quad
# F'(x) \approx \frac{1}{2\Delta x} \left(F\left(x + \Delta x\right) - F\left(x - \Delta x\right)\right).
# $$
# 
# **1.1** Consider another finite difference approximation defined as in $(1)$, this time with $N=3$, and
# 
# $$
# \begin{cases}
# \alpha_1 = -\frac{4}{23}, &\alpha_2 = -\frac{9}{17}, &\alpha_3 = \frac{275}{391} \\
# \beta_1 = -\frac{3}{2}, &\beta_2 = -\frac{1}{3}, &\beta_2 = \frac{4}{5}
# \end{cases}.
# $$
# 
# Investigate the accuracy of this approximation.
# 
# **[5 marks]**
# 
# %% [markdown]
# First we take the Taylor expansion of $F(x - \frac{3}{2}\Delta x)$, $F(x - \frac{1}{3}\Delta x)$, $F(x + \frac{4}{5}\Delta x)$ about $x$:
# 
# $$ F \left( x -\frac{3}{2}\Delta x \right) = F \left( x \right) - \frac{3}{2}\Delta x F' \left( x \right) + \frac{9\Delta x^2}{8} F'' \left( x \right) - \frac{9\Delta x^3}{16} F'''\left(x\right) + O(\Delta x ^4) $$
# 
# $$ F \left( x -\frac{1}{3}\Delta x \right) = F \left( x \right) - \frac{1}{3}\Delta x F' \left( x \right) + \frac{\Delta x^2}{18} F'' \left( x \right) - \frac{\Delta x^3}{162} F'''\left(x\right) + O(\Delta x ^4) $$
# 
# $$ F \left( x +\frac{4}{5}\Delta x \right) = F \left( x \right) + \frac{4}{5}\Delta x F' \left( x \right) + \frac{8\Delta x^2}{25} F'' \left( x \right) + \frac{32\Delta x^3}{375} F'''\left(x\right) + O(\Delta x ^4) $$
# 
# Then we substitute the Taylor expansion of $F(x - \frac{3}{2}\Delta x)$, $F(x - \frac{1}{3}\Delta x)$, and $F(x + \frac{4}{5}\Delta x)$ above into the finite difference approximation $D_{F3}(x)$, we have
# 
# \begin{align} D_{F3}(x) &=\frac{1}{\Delta x}\left( -\frac{4}{23} F \left( x -\frac{3}{2}\Delta x \right) - \frac{9}{17} F \left( x -\frac{1}{3}\Delta x \right) + \frac{275}{391} F \left( x +\frac{4}{5}\Delta x \right) \right)\\ &= \frac{1}{\Delta x}\left( \Delta x F' \left( x \right) +\frac{29\Delta x^3}{180} F'''\left(x\right) + O(\Delta x ^4)\right)\\ &=  F' \left( x \right) +  \frac{29\Delta x}{180} F''' \left( x \right) + O(\Delta x^3)\end{align}
# 
# 
# To get an approximation to $F'(x)$, we rearrange the above to make $F'(x)$ the subject :
# 
# $$ F'(x) = D_{F3}(x) - \frac{29\Delta x}{180} F''' \left( x \right) - O(\Delta x^3) = D_{F3}(x) - O(\Delta x^2)$$
# 
# The error is
# 
# $$ D_{F3}(x) - F' \left( x \right) = O(\Delta x^2), $$
# 
# therefore $D_{F3}(x)$ is second order accurate.
# 
# 
# %% [markdown]
# **1.2** For an arbitrary choice of $\beta_i$ values, what is the minimum number of points $N_{p}$ required to find an approximation $(1)$ which is at least $p$th order accurate?
# 
# *Hint:* consider the Taylor expansion of $F \left( x + \beta_i \Delta x \right)$ around $x$.
# 
# **[3 marks]**
# 
# For a sufficiently regular function $F$ and a small enough $\Delta x$, consider the Taylor expansion of $F(x + \beta_i \Delta x)$ about $x$:
# 
# 
# $$
#   F \left( x + \beta_i  \Delta x \right) = F \left( x \right) + \beta_i  \Delta x F' \left( x \right) + \frac{ \beta_i \Delta x^2}{2} F'' \left( x \right) + \frac{ \beta_i \Delta x^3}{6} F'''\left(x\right) + \dots
# $$
# 
# Then we multiply $\gamma_i$ on both sides of the Taylor expansion of $F(x + \beta_i \Delta x)$ about $x$:
# 
# $$\gamma_i F \left( x + \beta_i\Delta x \right) = \gamma_i F \left( x \right) + \gamma_i \beta_i\Delta x F' \left( x \right) + \frac{\gamma_i(\beta_i\Delta x)^2}{2} F'' \left( x \right) + \frac{\gamma_i(\beta_i\Delta x)^3}{6} F'''\left(x\right) + \dots$$
# 
# 
# Assume the minimum number of points $N_{p}$ required to find an approximation $(1)$ is N and it is at least $p$th order accurate.
# $$
#   F \left( x + \beta_1  \Delta x \right) = F \left( x \right) + \beta_1  \Delta x F' \left( x \right) + \frac{ \beta_1 \Delta x^2}{2} F'' \left( x \right) + \frac{ \beta_1 \Delta x^3}{6} F'''\left(x\right) + \dots$$
# 
# $$ F \left( x + \beta_2  \Delta x \right) = F \left( x \right) + \beta_2  \Delta x F' \left( x \right) + \frac{ \beta_2 \Delta x^2}{2} F'' \left( x \right) + \frac{ \beta_2 \Delta x^3}{6} F'''\left(x\right) + \dots
# $$
# 
# To find an approxiamtion to $F' \left( x \right)$, we rearrange the above to make $F'(x)$ the subject : 
# $$ F'(x) = \frac{1}{\beta_i \Delta x}\left( F \left( x +\beta_i\Delta x \right) - F \left( x \right) - \frac{(\beta_i\Delta x)^2}{2} F'' \left( x \right) - \frac{(\beta_i\Delta x)^3}{6} F'''  \left( x \right) - \dots\right) $$
# 
# We construct N's $\gamma$, for which  $\sum_{i = 1}^N\gamma_i $ equals to zero and  $\sum_{i = 1}^N\gamma_i\beta_i $ doesn't equals to zero.
# \begin{align}
# F' \left( x \right) = \frac{\sum_{i = 1}^N \gamma_i\frac{ F( x + \beta_i \Delta x)}{\Delta x}}{\sum_{i = 1}^N\gamma_i\beta_i} - \frac{\sum_{i = 1}^N\gamma_i F(x)}{\sum_{i = 1}^N\gamma_i\beta_i} -  \frac{\sum_{i = 1}^N \gamma_i\frac{ (\beta_i)^2 \Delta x)}{2}F''(x)}{\sum_{i = 1}^N\gamma_i\beta_i} - \frac{\sum_{i = 1}^N \gamma_i\frac{ (\beta_i)^3 (\Delta x)^2)}{6}F'''(x)}{\sum_{i = 1}^N\gamma_i\beta_i} - \dots\\
# =\sum_{i = 1}^N \alpha_i \frac{F \left( x + \beta_i \Delta x \right)}{\Delta x} -\sum_{i = 1}^N \alpha_i \frac{ (\beta_i)^2 \Delta x)}{2}F''(x) - \sum_{i = 1}^N \alpha_i \frac{ (\beta_i)^3 (\Delta x)^2)}{6}F'''(x) - \dots
# \end{align}
# 
# 
# 
# Given that it is at least $p$th order accurate, so we have:
# \begin{align}
# \sum_{i=1}^N\gamma_i &= 0\\
# \sum_{i=1}^N\gamma_i \beta_i& \neq 0\\
# \sum_{i=1}^N\gamma_i(\beta_i)^2 &= 0\\
# \dots\\
# \sum_{i=1}^N\gamma_i (\beta_i)^{p} &= 0
# \end{align}
# 
# So to satisfy these equations above, the number of $\gamma_i$  should be $p+1$ at least. If not, all $\gamma_i$ will be the same, which contradicts that $\beta_i$ can not be same with each other. Therefore, the minimum number of points $N_{p}$ is $p+1$.
# %% [markdown]
# ***📝 Discussion for question 1.2***
# %% [markdown]
# **1.3** Using your reasoning from **1.2**, write a function `FD_coefficients()` which, given $N_p$ values $\beta_i$, returns $N_p$ coefficients $\alpha_i$ such that the approximation $(1)$ is at least $p$th order accurate.
# 
# Use your function to obtain the coefficients $\alpha_i$ from **1.1**.
# 
# **[2 marks]**

# %%
import numpy as np
import math

def  FD_coefficients(beta_i):
    Np = len(beta_i)
    B = np.zeros([Np, Np])
    # Use loop to construct a linear equation
    for i in range (0, Np) :
        for j in range (0, Np) :
            B[j, i] = beta_i[i] ** j / math.factorial(j)
    b = np.zeros(Np)
    b[1] = 1
    # Solving the linear equation
    x = np.linalg.solve(B, b)
    return x
        
beta_i = np.array([-3/2, -1/3, 4/5])
alpha_i = FD_coefficients(beta_i)
print('The coefficients are',alpha_i)

# %% [markdown]
# ---
# ## Question 2: Root Finding [10 marks]
# 
# Consider the following polynomial of cubic order,
# 
# $$
# p(z) = z^3 + (c-1)z - c,
# $$
# where $c \in \mathbb{C}$.
# 
# This polynomial is complex differentiable, and we can apply Newton's method to find a complex root $z_\ast$, using a complex initial guess $z_0 = a_0 + ib_0$. In this problem, we seek to map the values of $z_0$ which lead to convergence to a root of $p$.
# 
# **2.1** Write a function `complex_newton(amin, amax, bmin, bmax, c, N, eps, target_roots)` which implements Newton's method to find roots of $p(z)$ using $N^2$ initial guesses $z_0 = a_0 + ib_0$. The input arguments are as follows:
# 
# - The real part $a_0$ of the initial guess should take `N` linearly spaced values between `amin` and `amax` (inclusive).
# - The imaginary part $b_0$ of the initial guess should take `N` linearly spaced values between `bmin` and `bmax` (inclusive).
# - `c` is the parameter $c \in \mathbb{C}$ in $p(z)$.
# - `eps` is the tolerance $\varepsilon > 0$.
# - `target_root` takes one of the following values:
#     - if `target_root` is given as `None`, then convergence should be considered as achieved if Newton's method has converged to any root of $p$.
#     - if `target_root` is given as a number, then convergence should only be considered as achieved if Newton's method has converged to the specific root $z_\ast =$ `target_root`.
# 
# Your function should return an array `kmax` of size $N \times N$, containing the total number of iterations required for convergence, for each value of $z_0$. You should decide what to do in case a particular value of $z_0$ doesn't lead to convergence.
#     
# Up to 2 marks will be given to solutions which iterate over each value of $z_0$. To obtain up to the full 4 marks, your implementation should be vectorised -- i.e. use a single loop to iterate Newton's method for all values of $z_0$ at once.
# 
# **[4 marks]**

# %%
import numpy as np

def complex_newton(amin, amax, bmin, bmax, c, N, eps, target_roots):
    # Check the parameter type
    assert isinstance (N, int), 'N should be an integer'
    # Check the parameter values    
    assert amin < amax, 'amin should be smaller than amax'
    assert bmin < bmax, 'bmin should be smaller than bmax'
    assert N > 0, 'N should be bigger than zero'
    assert eps > 0, 'eps should be bigger than zero'

    def F(z, c):
        return z ** 3 + (c-1) * z - c
    def Fp(z, c):
        return 3 * z ** 2 + c -1
    def G(z, c):
        return z - F(z, c) / Fp(z, c)
    kmax = np.ones([N, N])
    a0 = np.linspace(amin, amax, N)
    b0 = np.linspace(bmin, bmax, N)
    # Initialize z0
    z0 = []
    for i in range (0, N):
        for j in range (0, N):
            z0 = np.append(z0, complex(a0[i], b0[j]))
    if target_roots == "None":
        # Loop until convergence 
        for i in range (z0.shape[0]):
            its = 0
            z = z0[i]
            # Fixed point iteration
            while True:
                its += 1
                z_new = G(z, c)
                # Convergence achieved
                if abs(z_new - z) < eps or its > 50:
                    if abs(z_new - z) < eps:
                        kmax[i//N][i%N] = its
                    else:
                        kmax[i//N][i%N] = 0
                    break
                # Update value for next iteration
                z = z_new
        return kmax
    else:
        # Use a loop until convergence
        for i in range (z0.shape[0]):
            its = 0
            z = z0[i]
            while True:
                its += 1
                z_new = G(z, c)
                # Convergence achieved
                if abs(z_new - target_roots) < eps or its > 50:
                    if abs(z_new - target_roots) < eps:
                        kmax[i//N][i%N] = its
                    else:
                        kmax[i//N][i%N] = 0
                    break
                #Update for the next iteration
                z = z_new
        return kmax
 

# %% [markdown]
# **2.2** For $c = 0$, $a_0 \in [-5,5]$ and $b_0 \in [-5,5]$, with at least $N = 200$ values for each (you can increase $N$ if your computer allows it), use your function `complex_newton()` to calculate, for each $z_0 = a_0 + ib_0$, the total number of iterates needed to reach a disk of radius $\varepsilon$ around the root at $z = 1$. Present your results in a heatmap plot, with $a_0$ on the abscissa, $b_0$ on the ordinate and a colour map showing the total number of iterates. 
# 
# **[3 marks]**

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
a0 = np.linspace(-5, 5, 211)
b0 = np.linspace(-5, 5, 211)
cat =complex_newton(-5, 5, -5, 5, 0, 211, 1e-10, 1)
print (cat)
    
cat = pd.DataFrame(cat, columns = a0, index = b0)
    
fig = plt.figure()
sns_plot = sns.heatmap(cat, xticklabels = 50, yticklabels = 50)
sns_plot.tick_params(labelsize = 8)
plt.show()

# %% [markdown]
# **2.3** For $c = 0.32 + 1.64i$, map out the points $z_0$ for which Newton's method does not converge to any root. What does it do instead?
# 
# *Hint:* Pick a point $z_0$ in a region where Newton's method does not converge and print out, say, 50 iterates. What do you observe?
# 
# **[3 marks]**

# %%
# Initialization
c = 0.32 + 1.64j
N= 210 
a0 = np.linspace(-5, 5, 210)
b0 = np.linspace(-5, 5, 210)
z0 = np.zeros([N, N], dtype = complex)

for i in range(0,N) :
    for k in range(0,N):
        z0[i, k] = a0[i] + b0[k]*1j
r0 = z0[192, 35]
# Show the result 
print(' The point z0 is ', r0)
print(' The 50 iterates that using Newton method are')        
for k in range(50):
    #Apply  the Newton method
    r1 = (2 * r0 **3 + c) / (3 * r0 ** 2 + c - 1)
    print(r1)

# %% [markdown]
# ***📝 Discussion for question 2.3***
# %% [markdown]
# ---
# ## Question 3: Numerical Integration of an ODE [20 marks]
# 
# Cardiac tissue is an example of an excitable medium, where a small stimulus can lead to a large response (a heart beat). The FitzHugh-Nagumo model describes the electrical activity of a single cardiac cell in terms of the transmembrane potential $u$ and a recovery variable $v$
# 
# \begin{align}
#         \dot u & = f(u,v) = \frac{1}{\varepsilon} \left( u - \frac{u^3}{3} - v + I \right) \ , \\
#         \dot v & = g(u,v) = \varepsilon \left( u - \gamma v + \beta \right) \ ,
# \end{align}
# 
# where $I$ (a stimulus), $\varepsilon$, $\gamma$, and $\beta$ are known parameters.
# 
# The equation for $u$ leads to fast dynamics with the possibility of excitation, while the linear term proportional to $-v$ in the equation for the recovery variable produces slower dynamics and negative feedback. The FitzHugh-Nagumo model is an example of a stiff differential equation, where the stiffness becomes more pronounced for smaller $\varepsilon$.
# 
# In questions **3.1**, **3.2**, and **3.3**, we take $\varepsilon = 0.2$, $\gamma = 0.8$, and $\beta = 0.7$.
# 
# 
# **3.1** The fixed points, defined by $\dot u = f(u, v) = 0$ and $\dot v = g(u, v) = 0$, correspond to the state of a cell at rest. Write a function `resting_state()` to determine the values $(u_I^*, v_I^*)$ for the cell in its resting state for a given value of $I$ and a given initial guess $(u_{I, 0}, v_{I, 0})$, using Newton's method.
# 
# Use your function to compute $(u_I^*, v_I^*)$ for $I=0$ and $I = 0.5$, with initial guess $(u_{I, 0}, v_{I, 0}) = (0.2, 0.2)$.
# 
# 
# **[5 marks]**

# %%
import numpy as np
def resting_state(u0, v0, I, gamma):
    #Define a nonlinear system
    def F(u, v, I, gamma):
        return np.array([f(u, v, I), g(u, v, gamma)])

    def f(u, v, I):
        return 1. / eps * (u - u ** 3 / 3 - v + I)

    def g(u, v, gamma):
        return  (u - gamma * v + beta) * eps

    def Jac(u, v, gamma):
        #Initialize J
        J = np.zeros([2,2])
        #Define the jacobian matrix
        J[0, 0] = 1. / eps * (1 - u ** 2)
        J[1, 0] = eps
        J[0, 1] = -(1./eps)
        J[1, 1] = -eps * gamma
        return J

    # Create the intial guess
    x0 = np.array([u0, v0])
    x = x0
    #Use Newton's method 
    its =0
    while np.linalg.norm(F(x[0], x[1], I, gamma)) > 1e-10 and its <100:
        its = its +1
        e = -np.linalg.solve(Jac(x[0], x[1], gamma), F(x[0], x[1], I, gamma))
        x += e
    return x
#Set the value of eps and beta
eps = 0.2
beta = 0.7
#Show the result for I = 0 and I=0.5 with initial guess
print(f'If I =0, the value at resting state is {resting_state(0.2, 0.2, 0, 0.8)} with initial guess.')
print(f'If I =0.5, the value at resting state is {resting_state(0.2, 0.2, 0.5,0.8)} with initial guess.')

# %% [markdown]
# **3.2** Using the method of your choice **\***, compute the numerical solution $(u_n, v_n) \approx (u(n\Delta t), v(n\Delta t)), n=0, 1, 2, \dots$ for the FitzHugh-Nagumo model.
# 
# You should compute the solution for both $I = 0$ and $I = 0.5$, starting at time $t = 0$ until at least $t = 100$, with $(u_0 = 0.8, v_0 = 0.8)$ as the initial condition.
# 
# Present your results graphically by plotting
# 
# (a) $u_n$ and $v_n$ with **time** (not time step) on the x-axis,  
# (b) $v_n$ as a function of $u_n$. This will show what we call the solution trajectories in *phase space*.
# 
# You should format the plots so that the data presentation is clear and easy to understand.
# 
# Given what this mathematical model describes, and given that $I$ represents a stimulus, how do you interpret your results for the two different values of $I$? Describe your observations in less than 200 words.
# 
# 
# **\*** You may use e.g. the forward Euler method seen in Week 7 with a small enough time step, or use one of the functions provided by the `scipy.integrate` module, as seen in Quiz Q4.
# 
# 
# **[7 marks]**

# %%
# Define the initial parameters
delta_t = 0.01
t = np.linspace(0,100,int(100/delta_t))
x = np.zeros([2,len(t)])
x[:, 0] = np.array([0.8, 0.8])
# Compute the numericai solution by using forward Euler method
I = 0.0
for i in range(len(t)-1): 
    x[:, i+1] = x[:, i] + delta_t * f(x[:, i], I)
u = x[0, :]
v = x[1, :]

#plot the required figures
plt.plot(t, u, '-', label = 'u(t)')
plt.plot(t, v, '-', label = 'v(t)')
#The legend is located at upper right of the plot 
plt.legend( loc=1)
#Display the name for axis x and y 
plt.xlabel( 't')
plt.ylabel( 'value')
# Set the title
plt.title('graph of u(t) and v(t) when I = 0')
plt.show()

# %% [markdown]
# ***📝 Discussion for question 3.2***
# %% [markdown]
# **3.3** Compute the eigenvalues of the Jacobi matrix
#         
# $$
# \large
# \begin{pmatrix}
#     \frac{\partial f}{\partial u} & \frac{\partial f}{\partial v} \\ 
#     \frac{\partial g}{\partial u} & \frac{\partial g}{\partial v}
# \end{pmatrix}_{u = u_I^*, v = v_I^*}
# $$
# 
# evaluated at the fixed points $u = u_I^*, v = v_I^*$, for $I = 0$ and $I = 0.5$. What do you observe?
# 
# *You may use NumPy built-in functions to compute eigenvalues.*
# 
# 
# **[3 marks]**

# %%
import numpy as np

# Case 1: I = 0
uv_1 = resting_state(0.2, 0.2, 0, 0.8)
j_1 = Jac(uv_1[0], uv_1[1], 0.8)
eigenvalue_A = np.linalg.eig(j_1)[0]
# Display the value of eigenvalue
print('If I = 0, the eigenvalue will be', eigenvalue_A)

# Case 2: I = 0.5
uv_2 = resting_state(0.2, 0.2, 0.5, 0.8)
j_2 = Jac(uv_2[0], uv_2[1],0.8)
eigenvalue_B = np.linalg.eig(j_2)[0]
# Display the value of eigenvalue
print('If I = 0.5, the eigenvalue will be', eigenvalue_B)

# %% [markdown]
# ***📝 Discussion for question 3.3***
# %% [markdown]
# **3.4** For this question, we set $I = 0$ and $\gamma = 5$.
# 
# (a) Use the function `resting_state()` you wrote for **3.1** to find three fixed points, $(u_{(0)}^*, v_{(0)}^*)$, $(u_{(1)}^*, v_{(1)}^*)$ and $(u_{(2)}^*, v_{(2)}^*)$, using the initial conditions provided in the NumPy array `uv0` below (each row of the array constitutes a pair of initial conditions for one of the three fixed points).
# 
# (b) Compute the numerical solution $(u_n, v_n), n=0, 1, 2, \dots$ using the initial condition $(u_{(0)}^* + \delta, v_{(0)}^* + \delta)$, for $\delta \in \{0, 0.3, 0.6, 1.0\}$. This simulates the activity of a cell starting at a small perturbation $\delta$ of one of its resting states, in this case $(u_{(0)}^*, v_{(0)}^*)$.
# 
# Plot your results in a similar way as in question **3.2**, and discuss your observations in less than 150 words. In particular, does the solution always return to the same fixed point, i.e. the same resting state?
# 
# **[5 marks]**

# %%
import numpy as np

# Initial conditions
uv0 = np.array([[0.83928724, 0.64551717],
                [0.03831243, 0.43144263],
                [-1.7232432, -0.2604498]])

# %% [markdown]
# ***📝 Discussion for question 3.4***

