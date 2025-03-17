import numpy as np
import matplotlib.pyplot as plt

# Function to compute the tuning curve
def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * np.cos(np.radians(theta - theta_0)))

# Define the parameters
angles = np.linspace(-180, 180, 360)

# Anterior Thalamus parameters
params_thalamus = {"theta_0": 0, "A": 2.53, "B_expK": 34.8, "K": 8.08}
f_thalamus = tuning_curve(angles, **params_thalamus)

# Postsubiculum parameters
params_postsub = {"theta_0": 0, "A": 1.72, "B_expK": 94.8, "K": 5.2}
f_postsub = tuning_curve(angles, **params_postsub)

# Plot tuning curves
fig, axs = plt.subplots(1, 2, figsize=(24, 10))

# Cartesian Plot
axs[0].plot(angles, f_thalamus, label='Anterior Thalamus', linewidth=3, color='b')
axs[0].plot(angles, f_postsub, label='Postsubiculum', linewidth=3, color='r')
axs[0].set_xlabel("Head Direction (°)", fontsize=35)
axs[0].set_ylabel("Firing Rate (Hz)", fontsize=35)
axs[0].set_xticks([-180, -120, -60, 0, 60, 120, 180])
# axs[0].set_title("Directional Tuning Curves (Cartesian)")
axs[0].set_title("A", fontsize=35, pad=30)
axs[0].legend(fontsize=35, loc="upper right")
axs[0].tick_params(labelsize=35)
axs[0].grid(True)

# Polar Plot
axs[1].tick_params(labelsize=0)
axs[1].set_frame_on(False)
# ax.get_yaxis().set_visible(False)
ax_polar = fig.add_subplot(1, 2, 2, projection='polar')
ax_polar.plot(np.radians(angles), f_thalamus, linewidth=3, color='b')
ax_polar.plot(np.radians(angles), f_postsub, linewidth=3, color='r')
# ax_polar.set_title("Directional Tuning Curves (Polar)")
# ax_polar.legend(fontsize=40)
ax_polar.set_title("B", fontsize=35)
ax_polar.tick_params(labelsize=35)


plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Parameters from the paper (Figure 4A)
a = 6.34       # scaling factor
beta = 0.8
b = 10.0    # steepness parameter
c = 0.5        # bias

# Define the sigmoid function T(x) from Equation (4)
def T(x):
    return a * np.log(1 + np.exp(b * (x + c))) ** beta

# Choose an appropriate range of input current values (in arbitrary units)
# The range is chosen to display the rising part and saturation of the function.
x_values = np.linspace(-1, 1.5, 500)
y_values = T(x_values)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, lw=2, color='b')
plt.xlabel("Input current", fontsize=25)
plt.ylabel("Firing rate (Hz)", fontsize=25)
# plt.title("Sigmoid Function (Equation 4) - Figure 4A", fontsize=14)
plt.grid(True)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.legend(fontsize=25)
plt.show()









import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10  
t_max = 500
T = 50
t_eval = np.linspace(0, t_max, 500)  # Ensure enough time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
theta_deg = np.rad2deg(theta)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

# def inverse_custom_sigmoid(f):
#     return (1 / b) * np.log(np.exp((f / a) ** (1 / beta)) - 1) - c
def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    """
    Inverse of f = a * (log(1 + exp(b*(x + c))))^beta.
    Returns the array U such that sigmoid(U) = f.
    """
    # Step 1: y = f / a
    y = f / a
    
    # Step 2: take beta-th root
    y_pow = np.power(y, 1.0 / beta)
    
    # Step 3: exponentiate and subtract 1
    inside = np.exp(y_pow) - 1.0
    
    # Step 4: take log, divide by b, subtract c
    U = (1.0 / b) * np.log(inside) - c
    
    return U

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** (beta)

# Function to compute the tuning curve
def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Define the parameters for the tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}



# Compute the tuning curve
f_tuning = tuning_curve(theta, **params_thalamus)


# Synaptic weight components
def w_even(theta):
    return np.cos(theta)

def w_odd(theta, t):
    return np.sin(theta + 0.1 * t)

def calc_lambda(lambda_reg, f_hat):
    return lambda_reg * (np.max(np.abs(f_hat))** 2)

# Compute weight distribution using regularization
def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = np.fft.fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)  # Use the tuning curve here
    u_hat = np.fft.fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    w = np.fft.ifft(w_hat).real
    return np.fft.fft(w)


def compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg):
    # Compute weight distribution Fourier transform
    W_k = compute_weight_distribution(f_tuning, lambda_reg)

    # Get frequency components (corresponding to differentiation in Fourier domain)
    N = len(f_tuning)
    k = np.fft.fftfreq(N) * (2 * np.pi * N)  # Convert to proper frequency scaling
    
    # Compute the first derivative in Fourier domain
    W_derivative_k = 1j * k * W_k

    # Convert back to spatial domain
    w_derivative = np.fft.ifft(W_derivative_k).real
    gamma = -0.063
    return np.fft.fft(w_derivative * gamma)

def circular_convolution(w, f):
    conv = np.fft.ifft(w * np.fft.fft(f)).real
    # conv = np.fft.ifftshift(conv)
    return conv
    # return (np.fft.ifft(np.fft.fft(w) * np.fft.fft(f)).real) * dtheta

# ODE system
w_dist = 0
def du_dt(t, u):
    f_u = sigma(u)
    
    # Compute weight distribution dynamically 
    
    # Perform convolution with the computed weight distribution
    conv = circular_convolution(w_dist, f_u)
    
    return ((-u + conv) / T)


def plot_direction(lambda_reg):
    # Circular convolution
    # Set the initial condition 
    params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
    f_u0 = tuning_curve(theta, **params)
    u0 = inverse_custom_sigmoid(f_u0)
    
    # Solve ODE
    sol = solve_ivp(du_dt, [0, t_max], u0, t_eval=t_eval)
    
    # Compute firing rate
    firing_rate = sigma(sol.y)
    
    
    # Plot tuning curves
    fig, axs = plt.subplots(2, 2, figsize=(36, 30))
    
    # Cartesian Plot
    axs[0, 0].plot(theta_deg, firing_rate.T[0], linewidth=3, color='b')
    axs[0, 0].set_xlabel("Head Direction (°)", fontsize=40)
    axs[0, 0].set_ylabel("Firing Rate (Hz)", fontsize=40)
    axs[0, 0].set_title("t=0", fontsize=40)
    axs[0, 0].tick_params(labelsize=30)
    axs[0, 0].set_xticks([0, 90, 180, 270, 360])
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(theta_deg, firing_rate.T[130], linewidth=3, color='b')
    axs[0, 1].set_xlabel("Head Direction (°)", fontsize=40)
    axs[0, 1].set_ylabel("Firing Rate (Hz)", fontsize=40)
    axs[0, 1].set_title("t=130", fontsize=40)
    axs[0, 1].tick_params(labelsize=30)
    axs[0, 1].set_xticks([0, 90, 180, 270, 360])
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(theta_deg, firing_rate.T[260], linewidth=3, color='b')
    axs[1, 0].set_xlabel("Head Direction (°)", fontsize=40)
    axs[1, 0].set_ylabel("Firing Rate (Hz)", fontsize=40)
    axs[1, 0].set_title("t=260", fontsize=40)
    axs[1, 0].set_xticks([0, 90, 180, 270, 360])
    axs[1, 0].tick_params(labelsize=30)
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(theta_deg, firing_rate.T[400], linewidth=3, color='b')
    axs[1, 1].set_xlabel("Head Direction (°)", fontsize=40)
    axs[1, 1].set_ylabel("Firing Rate (Hz)", fontsize=40)
    axs[1, 1].set_title("t=400", fontsize=40)
    axs[1, 1].set_xticks([0, 90, 180, 270, 360])
    axs[1, 1].tick_params(labelsize=30)
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # fig.suptitle(f"Directional Tuning Curves for lambda_0 = {lambda_reg}")
    plt.show()
    
    # Create 3D plot
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Meshgrid for 3D plot
    T_grid, Theta_grid = np.meshgrid(t_eval, theta_deg)
    
    # Ensure shapes match
    firing_rate = firing_rate.reshape((num_points, len(t_eval)))
    
    # Plot the firing rate surface
    ax.plot_surface(Theta_grid, T_grid, firing_rate, cmap='plasma')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=30, labelpad=20)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_zticks([5, 15, 25, 35])
    ax.set_ylabel('Time (ms)', fontsize=30, labelpad=35)
    ax.set_zlabel('Firing Rate (Hz)', fontsize=30, labelpad=-30)
    ax.tick_params(labelsize=30)
    # ax.set_title('3D Evolution of Firing Rate σ(u(θ, t))')
    
    
    # Fine-tune z-label position
    # ax.zaxis.label.set_rotation(180)  # Rotate the z-label for better visibility
    # ax.zaxis.label.set_verticalalignment('top')  # Align label properl
    
    plt.show()
    
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(24, 14))
    
    # Plot heatmap using pcolormesh
    c = ax.pcolormesh(theta_deg, t_eval, firing_rate.T, cmap='plasma', shading='auto')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=60)
    ax.set_ylabel('Time (ms)', fontsize=60)
    ax.set_xticks([90, 180, 270, 360])
    ax.tick_params(labelsize=60)
    
    # Colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Firing Rate (Hz)', fontsize=60)
    cbar.ax.tick_params(labelsize=60)
    
    plt.show()
    
    
    # Plot weight distribution
    plt.figure(figsize=(12, 5))
    
    # Cartesian Plot
    plt.plot(theta, np.fft.ifft(w_dist).real, color='b')
    plt.xlabel("Head Direction (°)")
    plt.ylabel("Weight")
    plt.title("Weight Distribution")
    plt.legend()
    plt.grid(True)
    
    plt.show()

lambda_reg = 10**(-3)
w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
plot_direction(lambda_reg)

# alpha = 0.00201
alpha = 0.000201
w_dist = compute_weight_distribution(f_tuning, lambda_reg) + np.fft.fft(alpha * np.sin(theta))
plot_direction(lambda_reg)









# fig, axs = plt.subplots(2, 3, figsize=(30, 16))
fig, axs = plt.subplots(2, 3, figsize=(18, 10))


theta_angles = np.rad2deg(theta)



theta_angles = theta_angles - 180


w_dist = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
w_dist_even = w_dist * 10**3

w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
w_dist_odd = w_dist * 10**3


w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
w_dist_comb = w_dist * 10**3

axs[0, 0].plot(theta_angles, w_dist_even, color='b')
# axs[1, 0].set_title("λ=10^-2")
# axs[0, 0].set_xlabel("Head Direction (°)", fontsize=20)
axs[0, 0].set_ylabel("Weight", fontsize=22)
axs[0, 0].set_xticks([-180, -90, 0, 90, 180])
axs[0, 0].set_xticklabels([])
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].text(100, -0.05, 'W(θ)', fontsize=22)
axs[0, 0].tick_params(axis='both', which='major', labelsize=22)

axs[0, 1].plot(theta_angles, w_dist_odd, color='b')
# axs[0, 1].set_xlabel("Head Direction (°)", fontsize=20)
# axs[1, 1].set_title("λ=10^-3")
axs[0, 1].set_xticks([-180, -90, 0, 90, 180])
axs[0, 1].set_xticklabels([])
axs[0, 1].set_yticks([-0.5, 0.0, 0.5])
axs[0, 1].set_yticklabels([-0.5, 0.0, 0.5])
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].text(85, -0.35, "γW'(θ)", fontsize=22)
axs[0, 1].tick_params(axis='both', which='major', labelsize=22)

axs[0, 2].plot(theta_angles, w_dist_comb, color='b')
# axs[0, 2].set_xlabel("Head Direction (°)", fontsize=20)
# axs[1, 2].set_title("λ=10^-4")
axs[0, 2].set_xticks([-180, -90, 0, 90, 180])
axs[0, 2].set_xticklabels([])
axs[0, 2].set_yticks([-0.5, 0.0, 0.5])
axs[0, 2].set_yticklabels([-0.5, 0.0, 0.5])
axs[0, 2].legend()
axs[0, 2].grid(True)
axs[0, 2].text(-10, -0.42, "W(θ) + γW'(θ)", fontsize=22)
axs[0, 2].tick_params(axis='both', which='major', labelsize=22)



# alpha = 0.0000201
w_dist = alpha * np.sin(theta)
w_dist = np.roll(w_dist, -num_points // 2)
w_dist_odd = w_dist * 10**3


w_dist = np.fft.fft(alpha * np.sin(theta)) + compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
w_dist_comb = w_dist * 10**3

axs[1, 0].plot(theta_angles, w_dist_even, color='b')
# axs[1, 0].set_title("λ=10^-2")
axs[1, 0].set_xlabel("Head Direction (°)", fontsize=22)
axs[1, 0].set_ylabel("Weight", fontsize=22)
axs[1, 0].set_xticks([-180, -90, 0, 90, 180])
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].text(100, -0.05, 'W(θ)', fontsize=22)
axs[1, 0].tick_params(axis='both', which='major', labelsize=22)

axs[1, 1].plot(theta_angles, w_dist_odd, color='b')
axs[1, 1].set_xlabel("Head Direction (°)", fontsize=22)
# axs[1, 1].set_title("λ=10^-3")
axs[1, 1].set_xticks([-180, -90, 0, 90, 180])
axs[1, 1].set_yticks([-0.2, 0.0, 0.2])
axs[1, 1].set_yticklabels([-0.2, 0.0, 0.2])
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].text(80, -0.15, 'αsinθ', fontsize=22)
axs[1, 1].tick_params(axis='both', which='major', labelsize=22)

axs[1, 2].plot(theta_angles, w_dist_comb, color='b')
axs[1, 2].set_xlabel("Head Direction (°)", fontsize=22)
# axs[1, 2].set_title("λ=10^-4")
axs[1, 2].set_xticks([-180, -90, 0, 90, 180])
axs[1, 2].set_yticks([-0.2, 0.0, 0.2])
axs[1, 2].set_yticklabels([-0.2, 0.0, 0.2])
axs[1, 2].legend()
axs[1, 2].grid(True)
axs[1, 2].text(0, -0.27, 'W(θ) + αsinθ', fontsize=22)
axs[1, 2].tick_params(axis='both', which='major', labelsize=22)


plt.show()








import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10  
t_max = 500
T = 10
lambda_reg = 10**(-5)
t_eval = np.linspace(0, t_max, 500)  # Ensure enough time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

# def inverse_custom_sigmoid(f):
#     return (1 / b) * np.log(np.exp((f / a) ** (1 / beta)) - 1) - c
def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    """
    Inverse of f = a * (log(1 + exp(b*(x + c))))^beta.
    Returns the array U such that sigmoid(U) = f.
    """
    # Step 1: y = f / a
    y = f / a
    
    # Step 2: take beta-th root
    y_pow = np.power(y, 1.0 / beta)
    
    # Step 3: exponentiate and subtract 1
    inside = np.exp(y_pow) - 1.0
    
    # Step 4: take log, divide by b, subtract c
    U = (1.0 / b) * np.log(inside) - c
    
    return U

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** (beta)

# Function to compute the tuning curve
def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Define the parameters for the tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}



# Compute the tuning curve
f_tuning = tuning_curve(theta, **params_thalamus)

# Synaptic weight components
def w_even(theta):
    return np.cos(theta)

def w_odd(theta, t):
    return np.sin(theta + 0.1 * t)

def calc_lambda(lambda_reg, f_hat):
    return lambda_reg * (np.max(np.abs(f_hat))** 2)

# Compute weight distribution using regularization
def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = np.fft.fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)  # Use the tuning curve here
    u_hat = np.fft.fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    w = np.fft.ifft(w_hat).real
    return np.fft.fft(w)

# Circular convolution
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
def circular_convolution(w, f):
    conv = np.fft.ifft(w * np.fft.fft(f)).real
    # conv = np.fft.ifftshift(conv)
    return conv
    # return (np.fft.ifft(np.fft.fft(w) * np.fft.fft(f)).real) * dtheta

# ODE system
def du_dt(t, u):
    f_u = sigma(u)
    
    # Compute weight distribution dynamically 
    
    # Perform convolution with the computed weight distribution
    conv = circular_convolution(w_dist, f_u)
    
    return ((-u + conv) / T)
def steady(u):
    f_u = sigma(u)
    
    # Compute weight distribution dynamically 
    
    # Perform convolution with the computed weight distribution
    conv = circular_convolution(w_dist, f_u)
    
    return ((-u + conv) / T)

def fire():
    # Set the initial condition 
    u0 = inverse_custom_sigmoid(f_tuning) 
    y_steady = fsolve(steady, u0)

    # Solve ODE
    sol = solve_ivp(du_dt, [0, t_max], u0, t_eval=t_eval)

    # Compute firing rate
    firing_rate = sigma(sol.y)

    if lambda_reg == 10**(-4):
        return firing_rate.T[400]

    return firing_rate.T[200]

# Plot tuning curves
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

theta_angles = np.rad2deg(theta)

# Cartesian Plot
lambda_reg = 10**(-1)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
f1 = fire()
axs[0, 0].plot(theta_angles, f1, label='Actual', color='b')
axs[0, 0].plot(theta_angles, f_tuning, label='Desired', color='r')
axs[0, 0].set_xlabel("Head Direction (°)")
axs[0, 0].set_ylabel("Firing Rate (Hz)")
axs[0, 0].set_title("λ=10^-1")
axs[0, 0].set_xticks([0, 60, 120, 180, 240, 300, 360])
axs[0, 0].legend()
axs[0, 0].grid(True)

lambda_reg = 10**(-2)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
f2 = fire()
axs[0, 1].plot(theta_angles, f2, label='Actual', color='b')
axs[0, 1].plot(theta_angles, f_tuning, label='Desired', color='r')
axs[0, 1].set_xlabel("Head Direction (°)")
axs[0, 1].set_ylabel("Firing Rate (Hz)")
axs[0, 1].set_title("λ=10^-2")
axs[0, 1].set_xticks([0, 60, 120, 180, 240, 300, 360])
axs[0, 1].legend()
axs[0, 1].grid(True)

lambda_reg = 10**(-3)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
f3 = fire()
axs[1, 0].plot(theta_angles, f3, label='Actual', color='b')
axs[1, 0].plot(theta_angles, f_tuning, label='Desired', color='r')
axs[1, 0].set_xlabel("Head Direction (°)")
axs[1, 0].set_ylabel("Firing Rate (Hz)")
axs[1, 0].set_title("λ=10^-3")
axs[1, 0].set_xticks([0, 60, 120, 180, 240, 300, 360])
axs[1, 0].legend()
axs[1, 0].grid(True)

lambda_reg = 10**(-4)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
f4 = fire()
axs[1, 1].plot(theta_angles, f4, label='Actual', color='b')
axs[1, 1].plot(theta_angles, f_tuning, label='Desired', color='r')
axs[1, 1].set_xlabel("Head Direction (°)")
axs[1, 1].set_ylabel("Firing Rate (Hz)")
axs[1, 1].set_title("λ=10^-4")
axs[1, 1].set_xticks([0, 60, 120, 180, 240, 300, 360])
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.show()



plt.plot(figsize=(12, 10))

theta_angles = np.rad2deg(theta)

# Cartesian Plot
lambda_reg = 10**(-1)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
f1 = fire()
# plt.plot(theta_angles, f1, label='10^-1', color='r')
plt.plot(theta_angles, f2, label='10^-2', color='r')
plt.plot(theta_angles, f3, label='10^-3', color='black')
plt.plot(theta_angles, f4, label='10^-4', color='g')
plt.plot(theta_angles, f_tuning, label='Desired', color='b')
plt.xlabel("Head Direction (°)")
plt.ylabel("Firing Rate (Hz)")
plt.xticks([0, 60, 120, 180, 240, 300, 360])
plt.legend()
plt.grid(True)

plt.show()






fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Cartesian Plot
lambda_reg = 10**(-1)
theta_angles = theta_angles - 180
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
f1_w_dist = w_dist * 10**3

axs[0, 0].plot(theta_angles, f1_w_dist, color='b')
# axs[0, 0].set_xlabel("Head Direction (°)")
axs[0, 0].set_ylabel("Weight (10^3)")
axs[0, 0].set_title("λ=10^-1")
axs[0, 0].set_xticks([-180, -120, -60, 0, 60, 120, 180])
axs[0, 0].legend()
axs[0, 0].grid(True)

lambda_reg = 10**(-2)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
f2_w_dist = w_dist * 10**3

axs[0, 1].plot(theta_angles, f2_w_dist, color='b')
# axs[0, 1].set_xlabel("Head Direction (°)")
# axs[0, 1].set_ylabel("Weight")
axs[0, 1].set_title("λ=10^-2")
axs[0, 1].set_xticks([-180, -120, -60, 0, 60, 120, 180])
axs[0, 1].legend()
axs[0, 1].grid(True)

lambda_reg = 10**(-3)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
f3_w_dist = w_dist * 10**3

axs[1, 0].plot(theta_angles, f3_w_dist, color='b')
axs[1, 0].set_xlabel("Head Direction (°)")
axs[1, 0].set_ylabel("Weight (10^3)")
axs[1, 0].set_title("λ=10^-3")
axs[1, 0].set_xticks([-180, -120, -60, 0, 60, 120, 180])
axs[1, 0].legend()
axs[1, 0].grid(True)

lambda_reg = 10**(-4)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = np.fft.ifft(w_dist).real
w_dist = np.roll(w_dist, -num_points // 2)
f4_w_dist = w_dist * 10**3

axs[1, 1].plot(theta_angles, f4_w_dist, color='b')
axs[1, 1].set_xlabel("Head Direction (°)")
# axs[1, 1].set_ylabel("Weight")
axs[1, 1].set_title("λ=10^-4")
axs[1, 1].set_xticks([-180, -120, -60, 0, 60, 120, 180])
axs[1, 1].legend()
axs[1, 1].grid(True)








lambda_values = [10**(-1), 10**(-2), 10**(-3), 10**(-4)]
w_dist_values = [f1_w_dist, f2_w_dist, f3_w_dist, f4_w_dist]
z_values = np.arange(len(lambda_values))  # Different Z positions

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, lambda_reg in enumerate(lambda_values):
    w_dist = w_dist_values[i]

    ax.plot(theta_angles, w_dist, zs=z_values[i], zdir='y', label=f"10^{-(i+1)}")

# Labels and ticks
ax.set_xlabel("Head Direction (°)", labelpad=15, fontsize=20)
ax.set_zlabel("Weight", labelpad=35, fontsize=20)
ax.set_ylabel("λo", fontsize=20)

ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_zticks([-0.2, 0, 0.2, 0.4])
ax.set_yticks(z_values)
ax.set_yticklabels([f"10^{-(i+1)}" for i in range(len(lambda_values))])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='z', labelsize=15, pad=15)
ax.tick_params(axis='y', labelsize=0)

# ax.tick_params(axis='y', pad=10)
# ax.tick_params(axis='z', pad=10)

ax.legend(fontsize=20, loc="upper left")
ax.view_init(elev=20, azim=-90)  # Adjust view angle

plt.show()









fig, axs = plt.subplots(2, 3, figsize=(18, 10))


theta_angles = np.rad2deg(theta)

axs[0, 0].plot(theta_angles, f2, label='Actual', color='b')
axs[0, 0].plot(theta_angles, f_tuning, label='Desired', color='r')
# axs[0, 0].set_xlabel("Head Direction (°)")
axs[0, 0].set_ylabel("Firing Rate (Hz)", fontsize=22)
axs[0, 0].set_title(r"$\lambda=10^{-2}$", fontsize=22)
axs[0, 0].set_xticks([0, 90, 180, 270, 360])
axs[0, 0].legend(fontsize=22)
axs[0, 0].grid(True)
axs[0, 0].tick_params(axis='both', which='major', labelsize=22)

axs[0, 1].plot(theta_angles, f3, label='Actual', color='b')
axs[0, 1].plot(theta_angles, f_tuning, label='Desired', color='r')
# axs[0, 1].set_xlabel("Head Direction (°)")
# axs[0, 1].set_ylabel("Firing Rate (Hz)")
axs[0, 1].set_title(r"$\lambda=10^{-3}$", fontsize=22)
axs[0, 1].set_xticks([0, 90, 180, 270, 360])
axs[0, 1].legend(fontsize=22)
axs[0, 1].grid(True)
axs[0, 1].tick_params(axis='both', which='major', labelsize=22)

axs[0, 2].plot(theta_angles, f4, label='Actual', color='b')
axs[0, 2].plot(theta_angles, f_tuning, label='Desired', color='r')
# axs[0, 2].set_xlabel("Head Direction (°)")
# axs[0, 2].set_ylabel("Firing Rate (Hz)", fontsize=20)
axs[0, 2].set_title(r"$\lambda=10^{-4}$", fontsize=22)
axs[0, 2].set_xticks([0, 90, 180, 270, 360])
axs[0, 2].legend(fontsize=22)
axs[0, 2].grid(True)
axs[0, 2].tick_params(axis='both', which='major', labelsize=22)





theta_angles = theta_angles - 180
axs[1, 0].plot(theta_angles, f2_w_dist, color='b')
# axs[1, 0].set_title("λ=10^-2")
axs[1, 0].set_xlabel("Head Direction (°)", fontsize=22)
axs[1, 0].set_ylabel("Weight", fontsize=22)
axs[1, 0].set_xticks([-180, -90, 0, 90, 180])
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].tick_params(axis='both', which='major', labelsize=22)

axs[1, 1].plot(theta_angles, f3_w_dist, color='b')
axs[1, 1].set_xlabel("Head Direction (°)", fontsize=22)
# axs[1, 1].set_title("λ=10^-3")
axs[1, 1].set_xticks([-180, -90, 0, 90, 180])
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].tick_params(axis='both', which='major', labelsize=22)

axs[1, 2].plot(theta_angles, f4_w_dist, color='b')
axs[1, 2].set_xlabel("Head Direction (°)", fontsize=22)
# axs[1, 2].set_title("λ=10^-4")
axs[1, 2].set_xticks([-180, -90, 0, 90, 180])
axs[1, 2].legend()
axs[1, 2].grid(True)
axs[1, 2].tick_params(axis='both', which='major', labelsize=22)


plt.show()






import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ddeint import ddeint

# Parameters
num_points = 2**10  
t_max = 500
tau_u = 50  # Decay time constant
tau_delay = 10  # Delay parameter
t_eval = np.linspace(0, t_max, 500)  # Time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
theta_deg = np.rad2deg(theta)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    y = f / a
    y_pow = np.power(y, 1.0 / beta)
    inside = np.exp(y_pow) - 1.0
    return (1.0 / b) * np.log(inside) - c

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** beta

def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Parameters for tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}
f_tuning = tuning_curve(theta, **params_thalamus)

def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)
    u_hat = fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    return fft(ifft(w_hat).real)  # Enforce real values

def compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg):
    # Compute weight distribution Fourier transform
    W_k = compute_weight_distribution(f_tuning, lambda_reg)

    # Get frequency components (corresponding to differentiation in Fourier domain)
    N = len(f_tuning)
    k = np.fft.fftfreq(N) * (2 * np.pi * N)  # Convert to proper frequency scaling
    
    # Compute the first derivative in Fourier domain
    W_derivative_k = 1j * k * W_k

    # Convert back to spatial domain
    w_derivative = np.fft.ifft(W_derivative_k).real
    gamma = -0.063
    return np.fft.fft(w_derivative * gamma)

def circular_convolution(w, f):
    return ifft(w * fft(f)).real

params = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}
f_u0 = tuning_curve(theta, **params)
u0 = inverse_custom_sigmoid(f_u0)

# Define the delayed differential equation
def dde_system(U, t):
    u_t = U(t)
    u_t_tau = U(t - tau_delay) if t > tau_delay else u0  # Handle initial condition
    
    # f_u_t = sigma(u_t)
    f_u_t_tau = sigma(u_t_tau)
    
    # conv_t = circular_convolution(w_dist, f_u_t)
    conv_t_tau = circular_convolution(w_dist, f_u_t_tau)
    
    # return (-u_t + conv_t + conv_t_tau) / tau_u
    return (-u_t + conv_t_tau) / tau_u

def simulate():
    # Solve the delayed differential equation
    sol = ddeint(dde_system, lambda t: u0, t_eval)
    firing_rate = sigma(sol.T)
    
    # Plot results
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    time_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    
    for i, idx in enumerate(time_indices):
        row, col = divmod(i, 3)
        axs[row, col].plot(theta, firing_rate.T[idx], color='b')
        axs[row, col].set_xlabel("Head Direction (°)")
        axs[row, col].set_ylabel("Firing Rate (Hz)")
        axs[row, col].set_title(f"Tuning Curves at t={idx}")
        axs[row, col].grid(True)
    
    plt.suptitle(f"Delayed Directional Tuning Curves (λ={lambda_reg})")
    plt.show()
    
    # 3D plot
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Meshgrid for 3D plot
    T_grid, Theta_grid = np.meshgrid(t_eval, theta_deg)
    
    # Ensure shapes match
    firing_rate = firing_rate.reshape((num_points, len(t_eval)))
    
    # Plot the firing rate surface
    ax.plot_surface(Theta_grid, T_grid, firing_rate, cmap='plasma')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=25, labelpad=20)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylabel('Time (ms)', fontsize=25, labelpad=20)
    ax.set_zlabel('Firing Rate (Hz)', fontsize=25, labelpad=-30)
    ax.tick_params(labelsize=20)
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(24, 14))
    
    # Plot heatmap using pcolormesh
    c = ax.pcolormesh(theta_deg, t_eval, firing_rate.T, cmap='plasma', shading='auto')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=60)
    ax.set_ylabel('Time (ms)', fontsize=60)
    ax.set_xticks([90, 180, 270, 360])
    ax.tick_params(labelsize=60)
    
    # Colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Firing Rate (Hz)', fontsize=60)
    cbar.ax.tick_params(labelsize=60)
    
    plt.show()
    
lambda_reg = 10**(-3)
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
simulate()


params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
f_u0 = tuning_curve(theta, **params)
u0 = inverse_custom_sigmoid(f_u0)
w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
simulate()

alpha = 0.000201
w_dist = compute_weight_distribution(f_tuning, lambda_reg) + np.fft.fft(alpha * np.sin(theta))

simulate()





tau_delay = 70

 
w_dist = compute_weight_distribution(f_tuning, lambda_reg)
simulate()


params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
f_u0 = tuning_curve(theta, **params)
u0 = inverse_custom_sigmoid(f_u0)
w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
simulate()

alpha = 0.000201
w_dist = compute_weight_distribution(f_tuning, lambda_reg) + np.fft.fft(alpha * np.sin(theta))

simulate()






import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from ddeint import ddeint
from scipy.signal import welch
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10  
t_max = 500
tau_u = 50  # Decay time constant
tau_delay = 10  # Delay parameter
t_eval = np.linspace(0, t_max, 500)  # Time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
theta_deg = np.rad2deg(theta)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    y = f / a
    y_pow = np.power(y, 1.0 / beta)
    inside = np.exp(y_pow) - 1.0
    return (1.0 / b) * np.log(inside) - c

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** beta

def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Parameters for tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}
f_tuning = tuning_curve(theta, **params_thalamus)

def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)
    u_hat = fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    return fft(ifft(w_hat).real)  # Enforce real values

def compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg):
    # Compute weight distribution Fourier transform
    W_k = compute_weight_distribution(f_tuning, lambda_reg)

    # Get frequency components (corresponding to differentiation in Fourier domain)
    N = len(f_tuning)
    k = np.fft.fftfreq(N) * (2 * np.pi * N)  # Convert to proper frequency scaling
    
    # Compute the first derivative in Fourier domain
    W_derivative_k = 1j * k * W_k

    # Convert back to spatial domain
    w_derivative = np.fft.ifft(W_derivative_k).real
    gamma = -0.063
    return np.fft.fft(w_derivative * gamma)

def circular_convolution(w, f):
    return ifft(w * fft(f)).real

params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
f_u0 = tuning_curve(theta, **params)
u0 = inverse_custom_sigmoid(f_u0)

# Define the delayed differential equation
def dde_system(U, t):
    u_t = U(t)
    u_t_tau = U(t - tau_delay) if t > tau_delay else u0  # Handle initial condition
    
    # f_u_t = sigma(u_t)
    f_u_t_tau = sigma(u_t_tau)
    
    # conv_t = circular_convolution(w_dist, f_u_t)
    conv_t_tau = circular_convolution(w_dist, f_u_t_tau)
    
    # return (-u_t + conv_t + conv_t_tau) / tau_u
    return (-u_t + conv_t_tau) / tau_u

def simulate():
    # Solve the delayed differential equation
    sol = ddeint(dde_system, lambda t: u0, t_eval)
    firing_rate = sigma(sol.T)
    return (firing_rate.T, sol)
    
lambda_reg = 10**(-3)
w_even = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
w_odd = compute_weight_distribution(f_tuning, lambda_reg)
w_dist = w_even
dde_fire_even, dde_sol_even = simulate()

dde_max_f_even = []
dde_max_theta_even = []
for i in dde_fire_even:
    dde_max_f_even.append(np.max(i))
    dde_max_theta_even.append(theta[np.argmax(i)])

w_dist = w_odd
dde_fire_odd, dde_sol_odd = simulate()

dde_max_f_odd = []
dde_max_theta_odd = []
for i in dde_fire_odd:
    dde_max_f_odd.append(np.max(i))
    dde_max_theta_odd.append(theta[np.argmax(i)])
    

def du_dt(t, u):
    f_u = sigma(u)
    
    # Compute weight distribution dynamically 
    
    # Perform convolution with the computed weight distribution
    conv = circular_convolution(w_dist, f_u)
    
    return ((-u + conv) / tau_u)

def simulate():

    params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
    f_u0 = tuning_curve(theta, **params)
    u0 = inverse_custom_sigmoid(f_u0)

    # Solve ODE
    sol = solve_ivp(du_dt, [0, t_max], u0, t_eval=t_eval)

    # Compute firing rate
    firing_rate = sigma(sol.y)
    return (firing_rate.T, sol)

w_dist = w_even
ode_fire_even, ode_sol_even = simulate()

ode_max_f_even = []
ode_max_theta_even = []
for i in ode_fire_even:
    ode_max_f_even.append(np.max(i))
    ode_max_theta_even.append(theta[np.argmax(i)])

w_dist = w_odd
ode_fire_odd, ode_sol_odd = simulate()

ode_max_f_odd = []
ode_max_theta_odd = []
for i in ode_fire_odd:
    ode_max_f_odd.append(np.max(i))
    ode_max_theta_odd.append(theta[np.argmax(i)])
    
    


fig, axes = plt.subplots(1, 2, figsize=(30, 10))

c1 = axes[0].pcolormesh(theta_deg, t_eval, ode_fire_even, cmap='plasma', shading='auto')
axes[0].set_xlabel('θ (°)', fontsize=40)
axes[0].set_ylabel('Time (ms)', fontsize=30)
axes[0].set_title('A', fontsize=40)
axes[0].set_xticks([0, 90, 180, 270, 360])
axes[0].tick_params(labelsize=30)
cbar1 = fig.colorbar(c1, ax=axes[0])
cbar1.set_label('Firing Rate (Hz)', fontsize=40)
cbar1.ax.tick_params(labelsize=30)

# Plot second heatmap (firing_rate_even)
c2 = axes[1].pcolormesh(theta_deg, t_eval, dde_fire_even, cmap='plasma', shading='auto')
axes[1].set_xlabel('θ (°)', fontsize=40)
axes[1].set_ylabel('Time (ms)', fontsize=40)
axes[1].set_title('B', fontsize=40)
axes[1].set_xticks([0, 90, 180, 270, 360])
axes[1].tick_params(labelsize=30)
cbar2 = fig.colorbar(c2, ax=axes[1])
cbar2.set_label('Firing Rate (Hz)', fontsize=40)
cbar2.ax.tick_params(labelsize=30)

# Adjust layout for clarity
plt.tight_layout()
plt.show()




fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
axs[0].plot(t_eval, ode_max_f_odd, label="Original", color='b')
axs[0].plot(t_eval, dde_max_f_odd, label="DDE", color='r')
axs[0].set_xlabel("Time (ms)", fontsize=35)
axs[0].set_ylabel("Max Firing Rate (Hz)", fontsize=35)
# plt.title("ODE vs. DDE for static bumps", fontsize=30)
axs[0].set_title("A", fontsize=35)
axs[0].set_yticks([37, 38, 39, 40])
axs[0].legend(fontsize=35)
axs[0].tick_params(labelsize=30)
axs[0].grid(True)


axs[1].plot(t_eval, ode_max_f_even, label="Original", color='b')
axs[1].plot(t_eval, dde_max_f_even, label="DDE", color='r')
axs[1].set_xlabel("Time (ms)", fontsize=35)
axs[1].set_ylabel("Max Firing Rate (Hz)", fontsize=35)
# plt.title("ODE vs. DDE for travelling bumps", fontsize=30)
axs[1].set_title("B", fontsize=35)
axs[1].set_yticks([40, 50, 60])
axs[1].legend(fontsize=35)
axs[1].tick_params(labelsize=30)
axs[1].grid(True)





axs[2].plot(t_eval, np.rad2deg(ode_max_theta_even), label="Original", color='b')
axs[2].plot(t_eval, np.rad2deg(dde_max_theta_even), label="DDE", color='r')
axs[2].set_xlabel("Time (ms)", fontsize=35)
axs[2].set_ylabel("Angles", fontsize=35, labelpad=-5)
# plt.title("ODE vs. DDE for travelling bumps", fontsize=30)
axs[2].set_title("C", fontsize=35)
axs[2].set_yticks([60, 120, 180, 240, 300])
axs[2].legend(fontsize=35)
axs[2].tick_params(labelsize=30)
axs[2].grid(True)

plt.show()












# Compute Power Spectral Density (PSD) using Welch's method
# fs = len(t_eval) / t_max  # Sampling frequency

# f_ode, Pxx_ode = welch(ode_sol_odd.y[650], fs=fs, nperseg=256)
# f_dde, Pxx_dde = welch(dde_sol_odd.T[650], fs=fs, nperseg=256)

# # Plot PSD comparison
# plt.figure(figsize=(10, 6))
# plt.semilogy(f_ode, Pxx_ode, label="ODE", color='b')
# plt.semilogy(f_dde, Pxx_dde, label="DDE", color='r')
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Power Spectral Density")
# plt.title("Power Spectral Density of ODE vs. DDE")
# plt.legend()
# plt.grid(True)
# plt.show()



fig, axs = plt.subplots(1, 3, figsize=(30, 10))
# Compute Power Spectral Density (PSD) using Welch's method
fs = len(t_eval) / t_max  # Sampling frequency

f_ode, Pxx_ode = welch(ode_sol_even.y[450], fs=fs, nperseg=256)
f_dde, Pxx_dde = welch(dde_sol_even.T[450], fs=fs, nperseg=256)

# Plot PSD comparison
axs[0].semilogy(f_ode, Pxx_ode, label="Original", color='b')
axs[0].semilogy(f_dde, Pxx_dde, label="DDE", color='r')
axs[0].set_xlabel("Frequency (Hz)", fontsize=35)
axs[0].set_ylabel("Power Spectral Density", fontsize=35)
# plt.title("Power Spectral Density of ODE vs. DDE at time t=450")
# axs[0].set_title(f"Power Spectral Density of ODE vs. DDE at θ={round(np.rad2deg(theta[342]))}", fontsize=30)
axs[0].set_title("A", fontsize=35)
axs[0].legend(fontsize=35)
axs[0].tick_params(labelsize=30)
axs[0].grid(True)



fs = len(t_eval) / t_max  # Sampling frequency

f_ode, Pxx_ode = welch(ode_sol_even.y[650], fs=fs, nperseg=256)
f_dde, Pxx_dde = welch(dde_sol_even.T[650], fs=fs, nperseg=256)

# Plot PSD comparison
axs[1].semilogy(f_ode, Pxx_ode, label="Original", color='b')
axs[1].semilogy(f_dde, Pxx_dde, label="DDE", color='r')
axs[1].set_xlabel("Frequency (Hz)", fontsize=35)
# plt.ylabel("Power Spectral Density")
# axs[1].set_title(f"Power Spectral Density of ODE vs. DDE at θ={round(np.rad2deg(theta[683]))}")
axs[1].set_title("B", fontsize=35)
axs[1].legend(fontsize=35)
axs[1].tick_params(labelsize=30)
axs[1].grid(True)











fs = len(t_eval) / t_max  # Sampling frequency

f_ode_list = []
f_dde_list = []
Pxx_ode_list = []
Pxx_dde_list = []

for i in range(num_points):
    f_ode, Pxx_ode = welch(ode_sol_even.y[i], fs=fs, nperseg=256)
    f_dde, Pxx_dde = welch(dde_sol_even.T[i], fs=fs, nperseg=256)
    f_ode_list.append(f_ode)
    Pxx_ode_list.append(Pxx_ode)
    f_dde_list.append(f_dde)
    Pxx_dde_list.append(Pxx_dde)

f_ode = np.mean(f_ode_list, axis = 0)
f_dde = np.mean(f_dde_list, axis = 0)
Pxx_ode = np.mean(Pxx_ode_list, axis = 0)
Pxx_dde = np.mean(Pxx_dde_list, axis = 0)

# Plot PSD comparison
axs[2].semilogy(f_ode, Pxx_ode, label="Original", color='b')
axs[2].semilogy(f_dde, Pxx_dde, label="DDE", color='r')
axs[2].set_xlabel("Frequency (Hz)", fontsize=35)
# axs[2].set_ylabel("Power Spectral Density")
# axs[2].set_title("Power Spectral Density of ODE vs. DDE average across all angles")
axs[2].set_title("C", fontsize=35)
axs[2].legend(fontsize=35)
axs[2].tick_params(labelsize=30)
axs[2].grid(True)

plt.show()







import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10  
t_max = 500
tau_u = 50  # Decay time constant
tau_delay = 10  # Delay parameter
t_eval = np.linspace(0, t_max, 500)  # Ensure enough time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

# def inverse_custom_sigmoid(f):
#     return (1 / b) * np.log(np.exp((f / a) ** (1 / beta)) - 1) - c
def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    """
    Inverse of f = a * (log(1 + exp(b*(x + c))))^beta.
    Returns the array U such that sigmoid(U) = f.
    """
    # Step 1: y = f / a
    y = f / a
    
    # Step 2: take beta-th root
    y_pow = np.power(y, 1.0 / beta)
    
    # Step 3: exponentiate and subtract 1
    inside = np.exp(y_pow) - 1.0
    
    # Step 4: take log, divide by b, subtract c
    U = (1.0 / b) * np.log(inside) - c
    
    return U

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** (beta)

# Function to compute the tuning curve
def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Define the parameters for the tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}



# Compute the tuning curve
f_tuning = tuning_curve(theta, **params_thalamus)


# Synaptic weight components
def w_even(theta):
    return np.cos(theta)

def w_odd(theta, t):
    return np.sin(theta + 0.1 * t)

def calc_lambda(lambda_reg, f_hat):
    return lambda_reg * (np.max(np.abs(f_hat))** 2)

# Compute weight distribution using regularization
def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = np.fft.fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)  # Use the tuning curve here
    u_hat = np.fft.fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    w = np.fft.ifft(w_hat).real
    return np.fft.fft(w)


def compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg, gamma = -0.063):
    # Compute weight distribution Fourier transform
    W_k = compute_weight_distribution(f_tuning, lambda_reg)

    # Get frequency components (corresponding to differentiation in Fourier domain)
    N = len(f_tuning)
    k = np.fft.fftfreq(N) * (2 * np.pi * N)  # Convert to proper frequency scaling
    
    # Compute the first derivative in Fourier domain
    W_derivative_k = 1j * k * W_k

    # Convert back to spatial domain
    w_derivative = np.fft.ifft(W_derivative_k).real
    return np.fft.fft(w_derivative * gamma)

def circular_convolution(w, f):
    conv = np.fft.ifft(w * np.fft.fft(f)).real
    # conv = np.fft.ifftshift(conv)
    return conv
    # return (np.fft.ifft(np.fft.fft(w) * np.fft.fft(f)).real) * dtheta

# ODE system
w_dist = 0
def du_dt(t, u):
    f_u = sigma(u)
    
    # Compute weight distribution dynamically 
    
    # Perform convolution with the computed weight distribution
    conv = circular_convolution(w_dist, f_u)
    
    return ((-u + conv) / tau_u)


lambda_reg = 10**(-3)

def calc_speed(firing_rate, start = 100, end = 300):
    speeds = []
    for i in np.arange(start, end, tau_u):
        curr = np.rad2deg(theta[np.argmax(firing_rate[i + tau_u])] - theta[np.argmax(firing_rate[i])]) / (tau_u/1000)
        speeds.append(np.abs(curr))
    
    return np.mean(speeds)

def simulate():

    params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
    f_u0 = tuning_curve(theta, **params)
    u0 = inverse_custom_sigmoid(f_u0)

    # Solve ODE
    sol = solve_ivp(du_dt, [0, t_max], u0, t_eval=t_eval)

    # Compute firing rate
    firing_rate = sigma(sol.y)

    return calc_speed(firing_rate.T, 100, 300)

x_values = np.linspace(-0.01, -0.09, 100)
ode_y_values = []

for gamma in x_values:
    w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg, gamma) + compute_weight_distribution(f_tuning, lambda_reg)
    speed = simulate()
    ode_y_values.append(speed)

# Plot the function
params = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}
f_u0 = tuning_curve(theta, **params)
u0 = inverse_custom_sigmoid(f_u0)

# Define the delayed differential equation
def dde_system(U, t):
    u_t = U(t)
    u_t_tau = U(t - tau_delay) if t > tau_delay else u0  # Handle initial condition
    
    # f_u_t = sigma(u_t)
    f_u_t_tau = sigma(u_t_tau)
    
    # conv_t = circular_convolution(w_dist, f_u_t)
    conv_t_tau = circular_convolution(w_dist, f_u_t_tau)
    
    # return (-u_t + conv_t + conv_t_tau) / tau_u
    return (-u_t + conv_t_tau) / tau_u

def simulate():
    # Solve the delayed differential equation
    sol = ddeint(dde_system, lambda t: u0, t_eval)
    firing_rate = sigma(sol.T)
    
    return calc_speed(firing_rate.T, 100, 300)


dde_y_values = []
for gamma in x_values:
    w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg, gamma) + compute_weight_distribution(f_tuning, lambda_reg)
    speed = simulate()
    dde_y_values.append(speed)



plt.figure(figsize=(8, 6))
plt.plot(x_values, ode_y_values, lw=1, label="Original", color='b')
plt.plot(x_values, dde_y_values, lw=1, label="DDE", color='r')
plt.xlabel("γ", fontsize=25)
# plt.ylabel("Speed (deg/sec)", fontsize=30)
plt.ylabel("Speed (deg/sec)", fontsize=25)
# plt.title("Speed", fontsize=14)
plt.grid(True)
plt.legend(fontsize=25)
plt.tick_params(labelsize=25)
plt.show()






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10
t_max = 500
tau_u = 10
tau_a = 20
g = 1
lambda_reg = 10**(-3)
t_eval = np.linspace(0, t_max, 500)  # Ensure enough time points

theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
theta_deg = np.rad2deg(theta)
dtheta = 2 * np.pi / num_points

a, b, c, beta = 6.34, 10, 0.5, 0.8

def inverse_custom_sigmoid(f):
    return (1 / b) * np.log(np.exp((f / a) ** (1 / beta)) - 1) - c

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** beta

# Function to compute the tuning curve
def tuning_curve(theta, theta_0, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * np.cos(theta - theta_0))

# Define the parameters for the tuning curve
params_thalamus = {"theta_0": np.pi, "A": 1, "B_expK": 39, "K": 8}

# Compute the tuning curve
f_tuning = tuning_curve(theta, **params_thalamus)

# Compute weight distribution
def compute_weight_distribution(f_tuning, lambda_reg):
    f_hat = np.fft.fft(f_tuning)
    u = inverse_custom_sigmoid(f_tuning)  # Use the tuning curve here
    u_hat = np.fft.fft(u)
    fHatMax = np.max(np.abs(f_hat)) ** 2
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    w = np.fft.ifft(w_hat).real
    return np.fft.fft(w)

# Circular convolution
w_dist = compute_weight_distribution(f_tuning, lambda_reg)

def circular_convolution(w, f):
    return np.fft.ifft(w * np.fft.fft(f)).real

# ODE system
def system(t, y):
    u = y[:num_points]
    a = y[num_points:]
    
    f_u = sigma(u)
    psi = circular_convolution(w_dist, f_u)
    dudt = (-u + psi - (g * a)) / tau_u
    dadt = (u - a) / tau_a
    return np.concatenate([dudt, dadt])  # Concatenate derivatives

def plot():
    # Set the initial condition
    # u0 = inverse_custom_sigmoid(f_tuning)
    params = {"theta_0": np.pi / 3, "A": 1, "B_expK": 39, "K": 8}
    f_u0 = tuning_curve(theta, **params)
    u0 = inverse_custom_sigmoid(f_u0)
    a0 = np.full(num_points, 0)  # Ensure a0 has the same shape as u0
    y0 = np.concatenate([u0, a0])  # Ensure a single flat array
    
    # Solve ODE
    sol = solve_ivp(system, [0, t_max], y0, t_eval=t_eval)
    
    # Extract u and compute firing rate
    num_neurons = num_points  # Extract first num_points rows for u
    u_solution = sol.y[:num_neurons]  # Extract first num_neurons rows for u
    firing_rate = sigma(u_solution)
    
    
    # fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    
    # # Cartesian Plot
    # axs[0, 0].plot(theta, firing_rate.T[0], color='b')
    # axs[0, 0].set_xlabel("Head Direction (°)")
    # axs[0, 0].set_ylabel("Firing Rate (Hz)")
    # axs[0, 0].set_title("Directional Tuning Curves at t=0")
    # axs[0, 0].legend()
    # axs[0, 0].grid(True)
    
    # axs[0, 1].plot(theta, firing_rate.T[50], color='b')
    # axs[0, 1].set_xlabel("Head Direction (°)")
    # axs[0, 1].set_ylabel("Firing Rate (Hz)")
    # axs[0, 1].set_title("Directional Tuning Curves at t=50")
    # axs[0, 1].legend()
    # axs[0, 1].grid(True)
    
    # axs[0, 2].plot(theta, firing_rate.T[100], color='b')
    # axs[0, 2].set_xlabel("Head Direction (°)")
    # axs[0, 2].set_ylabel("Firing Rate (Hz)")
    # axs[0, 2].set_title("Directional Tuning Curves at t=100")
    # axs[0, 2].legend()
    # axs[0, 2].grid(True)
    
    # axs[1, 0].plot(theta, firing_rate.T[150], color='b')
    # axs[1, 0].set_xlabel("Head Direction (°)")
    # axs[1, 0].set_ylabel("Firing Rate (Hz)")
    # axs[1, 0].set_title("Directional Tuning Curves at t=150")
    # axs[1, 0].legend()
    # axs[1, 0].grid(True)
    
    # axs[1, 1].plot(theta, firing_rate.T[200], color='b')
    # axs[1, 1].set_xlabel("Head Direction (°)")
    # axs[1, 1].set_ylabel("Firing Rate (Hz)")
    # axs[1, 1].set_title("Directional Tuning Curves at t=200")
    # axs[1, 1].legend()
    # axs[1, 1].grid(True)
    
    # axs[1, 2].plot(theta, firing_rate.T[250], color='b')
    # axs[1, 2].set_xlabel("Head Direction (°)")
    # axs[1, 2].set_ylabel("Firing Rate (Hz)")
    # axs[1, 2].set_title("Directional Tuning Curves at t=250")
    # axs[1, 2].legend()
    # axs[1, 2].grid(True)
    
    # axs[2, 0].plot(theta, firing_rate.T[300], color='b')
    # axs[2, 0].set_xlabel("Head Direction (°)")
    # axs[2, 0].set_ylabel("Firing Rate (Hz)")
    # axs[2, 0].set_title("Directional Tuning Curves at t=300")
    # axs[2, 0].legend()
    # axs[2, 0].grid(True)
    
    # axs[2, 1].plot(theta, firing_rate.T[350], color='b')
    # axs[2, 1].set_xlabel("Head Direction (°)")
    # axs[2, 1].set_ylabel("Firing Rate (Hz)")
    # axs[2, 1].set_title("Directional Tuning Curves at t=350")
    # axs[2, 1].legend()
    # axs[2, 1].grid(True)
    
    # axs[2, 2].plot(theta, firing_rate.T[400], color='b')
    # axs[2, 2].set_xlabel("Head Direction (°)")
    # axs[2, 2].set_ylabel("Firing Rate (Hz)")
    # axs[2, 2].set_title("Directional Tuning Curves at t=400")
    # axs[2, 2].legend()
    # axs[2, 2].grid(True)
    
    # fig.suptitle(f"Directional Tuning Curves for tau_a = {tau_a}")
    # plt.show()
    
    
    
    
    
    # Create 3D plot
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Meshgrid for 3D plot
    T_grid, Theta_grid = np.meshgrid(t_eval, np.rad2deg(theta))
    
    # Plot the firing rate surface
    ax.plot_surface(Theta_grid, T_grid, firing_rate, cmap='plasma')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=30, labelpad=25)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_zticks([5, 15, 25, 35])
    ax.tick_params(labelsize=30)
    ax.set_ylabel('Time (ms)', fontsize=30, labelpad=25)
    ax.set_zlabel('Firing Rate (Hz)', fontsize=30, labelpad=-30)
    # ax.set_title(f'3D Evolution of Firing Rate σ(u(θ, t)) for tau_a = {tau_a}')
    
    plt.show()
    
    
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(24, 14))
    
    # Plot heatmap using pcolormesh
    c = ax.pcolormesh(theta_deg, t_eval, firing_rate.T, cmap='plasma', shading='auto')
    
    # Labels and title
    ax.set_xlabel('θ (°)', fontsize=60)
    ax.set_ylabel('Time (ms)', fontsize=60)
    ax.set_xticks([90, 180, 270, 360])
    ax.tick_params(labelsize=60)
    
    # Colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Firing Rate (Hz)', fontsize=60)
    cbar.ax.tick_params(labelsize=60)
    
    plt.show()
    
    
    
    
    
    
    # # Create figure with 2 subplots (1 row, 2 columns)
    # fig = plt.figure(figsize=(30, 10))
    # gs = GridSpec(1, 2, width_ratios=[1, 1])  # Ensures equal widths
    
    # # 3D Plot (Left)
    # ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 cols, first subplot
    
    # # Meshgrid for 3D plot
    # T_grid, Theta_grid = np.meshgrid(t_eval, theta)
    
    # # Plot the firing rate surface
    # ax1.plot_surface(Theta_grid, T_grid, firing_rate, cmap='plasma')
    
    # # Labels and title
    # ax1.set_xlabel('θ (Angle)', fontsize=20)
    # ax1.set_ylabel('Time (t)', fontsize=20)
    # ax1.set_zlabel('Firing Rate σ(u)', fontsize=20)
    # ax1.set_title(f'3D Evolution of Firing Rate σ(u(θ, t)) for tau_a = {tau_a}', fontsize=22)
    
    # # Heatmap (Right)
    # ax2 = fig.add_subplot(122)  # 1 row, 2 cols, second subplot
    
    # # Plot heatmap using pcolormesh
    # c = ax2.pcolormesh(theta_deg, t_eval, firing_rate.T, cmap='plasma', shading='auto')
    
    # # Labels and title
    # ax2.set_xlabel('θ (°)', fontsize=30)
    # ax2.set_ylabel('Time (ms)', fontsize=30)
    # ax2.set_xticks([0, 90, 180, 270, 360])
    # ax2.tick_params(labelsize=20)
    # ax2.set_title('Heatmap of Firing Rate', fontsize=22)
    
    # # Colorbar
    # cbar = fig.colorbar(c, ax=ax2)
    # cbar.set_label('Firing Rate (Hz)', fontsize=30)
    # cbar.ax.tick_params(labelsize=20)
    
    # # Adjust layout to prevent overlap
    # plt.tight_layout()
    
    # plt.show()
    
# plot()

tau_a = 50
# plot()

g = 2
# tau_a = 40
plot()






import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Parameters
num_points = 2**10  
t_max = 500
T = 50
t_eval = np.linspace(0, t_max, 500)  # Ensure enough time points

x = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
y = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
theta_x, theta_y = np.meshgrid(x, y)
dtheta = (2 * np.pi / num_points) ** 2  # Updated for 2D

a, b, c, beta = 6.34, 10, 0.5, 0.8

def inverse_custom_sigmoid(f, a=a, b=b, beta=beta, c=c):
    """
    Inverse of f = a * (log(1 + exp(b*(x + c))))^beta.
    Returns the array U such that sigmoid(U) = f.
    """
    # Step 1: y = f / a
    y = f / a
    
    # Step 2: take beta-th root
    y_pow = np.power(y, 1.0 / beta)
    
    # Step 3: exponentiate and subtract 1
    inside = np.exp(y_pow) - 1.0
    
    # Step 4: take log, divide by b, subtract c
    U = (1.0 / b) * np.log(inside) - c
    
    return U

def sigma(u):
    return a * np.log(1 + np.exp(b * (u + c))) ** (beta)

# Function to compute the tuning curve
def tuning_curve(theta_x, theta_y, theta_0x, theta_0y, A, B_expK, K):
    B = B_expK / np.exp(K)  # Solve for B
    return A + B * np.exp(K * (np.cos(theta_x - theta_0x) + np.cos(theta_y - theta_0y)) / 2)

# Define the parameters for the tuning curve
params_thalamus = {"theta_0x": np.pi, "theta_0y": np.pi, "A": 1, "B_expK": 39, "K": 8}

# Compute the tuning curve
f_tuning = tuning_curve(theta_x, theta_y, **params_thalamus)


# Plot tuning curves
plt.figure(figsize=(12, 5))

# Cartesian Plot
plt.figure(figsize=(12, 5))
plt.imshow(f_tuning, extent=[0, 2*np.pi, 0, 2*np.pi])
plt.colorbar(label='Firing Rate (Hz)')
plt.xlabel("Head Direction X (°)")
plt.ylabel("Head Direction Y (°)")
plt.title("Directional Tuning Curves (2D)")
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create coordinate matrices for plotting
X, Y = np.meshgrid(np.linspace(0, 2*np.pi, num_points), 
                   np.linspace(0, 2*np.pi, num_points))

# Plot the surface
surf = ax.plot_surface(X, Y, f_tuning, cmap='viridis')

# Add colorbar
# fig.colorbar(surf, label='Firing Rate (Hz)')

# Set labels and title
ax.set_xlabel('Head Direction X')
ax.set_ylabel('Head Direction Y')
ax.set_zlabel('Firing Rate (Hz)')
ax.set_title('Directional Tuning Curves (3D)')

# Adjust viewing angle for better visualization
ax.view_init(elev=30, azim=45)

plt.show()

# Synaptic weight components
def compute_weight_distribution(f_tuning, lambda_reg):
    # Compute the 2D Fourier transform of the tuning curve
    f_hat = np.fft.fft2(f_tuning)
    
    # Apply the inverse custom sigmoid to the 2D tuning curve
    u = inverse_custom_sigmoid(f_tuning)  # Ensure this function handles 2D arrays if needed
    u_hat = np.fft.fft2(u)
    
    # Compute the maximum power across all frequency components
    fHatMax = np.max(np.abs(f_hat)) ** 2
    
    # Calculate the weight distribution in Fourier space
    w_hat = (u_hat * f_hat) / (lambda_reg * fHatMax + np.abs(f_hat)**2)
    return w_hat

def compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg):
    W_k = compute_weight_distribution(f_tuning, lambda_reg) 

    # For a square array, get its dimension
    N = f_tuning.shape[0]  # assuming shape is (N, N)

    kx = np.fft.fftfreq(N) * (2 * np.pi * N)
    # Reshape to (1, N) so that it multiplies each column appropriately
    kx = kx.reshape(1, N)

    # Multiply the Fourier transform by 1j * kx to compute the derivative in Fourier domain (along x)
    W_derivative_k = 1j * kx * W_k
    w_derivative = np.fft.ifft2(W_derivative_k).real

    gamma = -0.063

    # Multiply by gamma and take the forward 2D FFT before returning.
    return np.fft.fft2(w_derivative * gamma)
    
def circular_convolution(w, f):
    # Perform 2D FFT convolution
    conv = np.fft.ifft2(w * np.fft.fft2(f)).real
    return conv


# ODE system
def du_dt(t, u):
    # Determine grid dimensions (assumes a square grid)
    N = int(np.sqrt(u.size))
    
    # Reshape the 1D state vector back to a 2D array
    u_2d = u.reshape((N, N))
    
    # Compute the firing rate (sigma should handle 2D arrays)
    f_u = sigma(u_2d)
    
    # Perform the convolution with the weight distribution (w_dist is 2D)
    conv = circular_convolution(w_dist, f_u)
    
    # Compute the time derivative in 2D
    du_2d = (-u_2d + conv) / T
    
    # Flatten the derivative to match the solver's expectation
    return du_2d.flatten()

def plot_sim(sol_t, t):
    u = sol_t[t]
    N = int(np.sqrt(u.size))
    # Reshape the 1D state vector back to a 2D array
    u_2d = u.reshape((N, N))
    
    
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create coordinate matrices for plotting
    # X, Y = np.meshgrid(np.linspace(0, 2*np.pi, num_points), 
    #                    np.linspace(0, 2*np.pi, num_points))
    X, Y = np.meshgrid(np.linspace(-180, 180, num_points), 
                       np.linspace(-180, 180, num_points))

    # Plot the surface
    firing_rate = sigma(u_2d)
    surf = ax.plot_surface(X, Y, firing_rate, cmap='viridis')

    # Add colorbar
    # fig.colorbar(surf, label='Firing Rate (Hz)')

    # Set labels and title
    ax.set_xlabel('X (°)', fontsize=32, labelpad=20)
    ax.set_xticks([-180, 0, 180],labels=[-180, 0, ''])
    ax.set_yticks([-180, 0, 180])
    ax.set_ylabel('Y (°)', fontsize=32, labelpad=20)
    ax.tick_params(labelsize=32)
    ax.set_zlabel('Firing Rate (Hz)', fontsize=32, labelpad=-30)
    # ax.set_title(f'time={t}ms', fontsize=25)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=45)

    plt.show()

def plot_direction():
    # Set the initial condition 
    params = {"theta_0x": np.pi / 3, "theta_0y": 4, "A": 1, "B_expK": 39, "K": 8}
    f_u0 = tuning_curve(theta_x, theta_y, **params)
    u0 = inverse_custom_sigmoid(f_u0).flatten()  # Flatten for solve_ivp
    
    # Solve ODE
    sol = solve_ivp(du_dt, [0, t_max], u0, t_eval=t_eval)
    
    sol_t = sol.y.T
    
    plot_sim(sol_t, 0)
    # plot_sim(sol_t, 100)
    # plot_sim(sol_t, 150)
    # plot_sim(sol_t, 200)
    # plot_sim(sol_t, 250)
    plot_sim(sol_t, 300)
    # plot_sim(sol_t, 350)
    # plot_sim(sol_t, 400)
    # plot_sim(sol_t, 450)
    plot_sim(sol_t, 499)
    

lambda_reg = 10**(-3)
w_dist = compute_gamma_weight_distribution_derivative(f_tuning, lambda_reg) + compute_weight_distribution(f_tuning, lambda_reg)
plot_direction()
