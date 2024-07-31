import numpy as np

def complicated_function(x, y):
    term1 = 0.5 + 0.5 * np.log(x**2 + 1.1 + y**4) * np.sin(3.0 * np.pi * x) * np.cos(1.2 * np.pi * y)
    term2 = 0.1 * np.sin(0.7 * np.pi * x**2)**2 * np.cos(1.2 * np.pi * y**2)**2 + 2.0
    term3 = -0.2 * np.exp(-(x**2 + y**2)) * np.cos(5 * np.pi * x * y)
    term4 = 0.1 * x**3 * y**3
    return term1 + np.log(term2) + term3 + term4

# Test the function with some values
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
z1 = complicated_function(X, Y)
z1_max, z2_min = np.max(z1), np.min(z2)
print(z1_max, z2_min)

def quad_function(x, y):
    term1 = 1.2 * (x-0.5) * (x-0.5);
    term2 = 0.4 * y * y
    return term1 + term2

z2 = quad_function(X, Y)
z2_max, z2_min = np.max(z2), np.min(z2)
z2 = (z2 - z2_min) / (z2_max - z2_min) * (z1_max - z1_min) + z1_min
z = z1 + z2


# Plotting the function to visualize it
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Complicated Function f(x, y) Bounded in [0, 1]')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('toy_plot_sum.png')

#######======Fit for z directly======########



#######======Fit for z1 directly======########
