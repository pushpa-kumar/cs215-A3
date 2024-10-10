import numpy as np
import matplotlib.pyplot as plt

# Custom Epanechnikov KDE class
class EpanechnikovKDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None

    def fit(self, data):
        """Fit the KDE model with the given data."""
        self.data=np.array(data)

    def epanechnikov_kernel(self, x, xi):
        """Epanechnikov kernel function."""
        i=(x-xi)/self.bandwidth
        distance_from_origin = np.linalg.norm(i)
        if distance_from_origin<=1:
            return (3/4) * (1 - distance_from_origin**2)
        return 0
        # return np.where(np.abs(i) <= 1, (3/4) * (1 - i**2), 0)
        
    def evaluate(self, x):
        """Evaluate the KDE at multiple points x.
        
        x should be a 2D array where each row is a point (x, y).
        """
        n = len(self.data)
        # Compute kernel density for each point in x, summing across all data points
        densities = np.array([np.sum([self.epanechnikov_kernel(xi, x_i) for x_i in self.data]) for xi in x])
        return (1 / (n * self.bandwidth)) * densities



# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kde=EpanechnikovKDE(bandwidth=1.0)

# TODO: Fit the data
kde.fit(data)

# TODO: Plot the estimated density in a 3D plot

x_min,x_max=data[:,0].min(),data[:,0].max()
y_min,y_max=data[:,1].min(),data[:,1].max()

x_range = np.linspace(x_min, x_max, 75)
y_range = np.linspace(y_min, y_max, 75)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Create a 2D array of (x, y) coordinate pairs
xy_stack = np.column_stack([x_grid.ravel(), y_grid.ravel()])
print(xy_stack.shape)

# Evaluate kde for all (x, y) pairs
z_values = kde.evaluate(xy_stack)

# Reshape the output to match the shape of x_grid or y_grid
z_grid = z_values.reshape(x_grid.shape)



fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
plt.title('Estimated Distribution of Transactions')
save_path = '../images/2/transaction distribution.png' 
plt.savefig(save_path) 
# plt.show()
# TODO: Save the plot 


