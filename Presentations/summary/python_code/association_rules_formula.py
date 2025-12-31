import numpy as np
import matplotlib.pyplot as plt
# Define the range of the variable 'd'
d = np.arange(0, 21, 1) # Integers from 0 to 20
# Define the function f(d) for the number of possible association rules
def f(d):
    return 3**d - 2**(d+1) + 1
# Compute the y-values for the defined integer points
y = f(d)
# Interpolate between points for a smooth curve
d_smooth = np.linspace(0, 20, 500) # Generate 500 points between 0 and 20
y_smooth = f(d_smooth)
# Plot the function
plt.figure(figsize=(10, 6)) # Set the figure size
plt.plot(d_smooth, y_smooth, label=r'$f(d) = 3^d - 2^{d+1} + 1$', color='blue', linewidth=2) 
plt.scatter(d, y, color='red', label='Integer Points', zorder=5) # Overlay integer points
# Additional plot settings
plt.title("Number of Possible Association Rules", fontsize=14)
plt.xlabel("Number of Unique Items (d)", fontsize=12)
plt.ylabel("Number of Possible Rules (f(d))", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6) # Add grid lines
plt.xticks(ticks=np.arange(0, 21, 1))
plt.legend(fontsize=12) # Show legend
plt.tight_layout() # Adjust layout for better appearance
# Show the plot
plt.show()