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
        return np.where(np.abs(i) <= 1, (3/4) * (1 - i**2), 0)
        
    def evaluate(self, x):
        """Evaluate the KDE at point x."""
        n=len(self.data)
        density=0
        for x_i in self.data:
            density+=self.epanechnikov_kernel(x,x_i)
        return (1/(n*self.bandwidth))*density


# Load the data from the NPZ file
data_file = np.load('transaction_data.npz')
data = data_file['data']

# TODO: Initialize the EpanechnikovKDE class
kde=EpanechnikovKDE(bandwidth=1.0)

# TODO: Fit the data
kde.fit(data)

# TODO: Plot the estimated density in a 3D plot
# x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
# y_vals = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
# x,y=np.meshgrid(x_vals,y_vals)
# z=np.zeros_like(x)
# for i in range(x.shape[0]):

x_min,x_max=data[:,0].min(),data[:,0].max()
y_min,y_max=data[:,1].min(),data[:,1].max()
x_range=np.linspace(x_min,x_max,50)
y_range=np.linspace(y_min,y_max,50)
x_grid,y_grid=np.meshgrid(x_range,y_range)
z_grid = np.zeros((x_grid.shape[0], x_grid.shape[1]))
for i in range(x_grid.shape[0]):
    for j in range(x_grid.shape[1]):
        x = x_grid[i, j]
        y = y_grid[i, j]
        z_grid[i, j] = kde.evaluate([x, y])[0]

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
plt.title('Estimated Distribution of Transactions')
save_path = '../images/2/transaction distribution.png' 
plt.savefig(save_path) 
# TODO: Save the plot 

