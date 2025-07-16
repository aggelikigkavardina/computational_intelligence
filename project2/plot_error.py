# Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
import pandas as pd
import matplotlib.pyplot as plt

# Load the error_data CSV file
try:
    error_data = pd.read_csv("error_data.csv")
except FileNotFoundError:
    print("Error: 'error_data.csv' file not found.")
    exit()

# Create the line plot
plt.figure(figsize=(8, 6))
plt.plot(error_data['M'], error_data['Error'], marker='o', linestyle='-', label='Error')

plt.xlabel("Number of Clusters (M)")
plt.ylabel("Best Error")
plt.title("K-means Clustering: Error vs. Number of Clusters")
plt.legend()
plt.grid(True) 

plt.show()