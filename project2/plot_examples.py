# Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
import matplotlib.pyplot as plt
import numpy as np

# Load examples file
data = np.loadtxt("examples")

x = data[:, 0]
y = data[:, 1]

# Create the plot
plt.figure()
plt.plot(x, y, '+')  # 'o' for points
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Plot of Generated Data")
plt.grid(True)

plt.show()
