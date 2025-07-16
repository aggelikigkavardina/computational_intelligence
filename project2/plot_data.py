# Alexandros Kokkinos 4084, Euaggelos Tempelopoulos 4175, Aggeliki Gkavardina 4042
import pandas as pd
import matplotlib.pyplot as plt

# Load the plot_data CSV file
data = pd.read_csv("plot_data.csv")

# M values and corresponding colors
M_values = [4, 6, 8, 10, 12] 
colors = ['red', 'blue', 'green', 'cyan', 'magenta'] 

# Counter for M values
M_counter = 0
current_M = M_values[M_counter]

plt.figure(figsize=(10, 6))

# Plot examples in black
plt.scatter(data[data['Type'] == '+']['X'], data[data['Type'] == '+']['Y'], marker='+', color='black', s=20)

# Plot centers with colors
for i, row in data.iterrows():
    if row['Type'] == '*':
        if current_M > 0:
            current_M -= 1
        else:
            M_counter += 1
            current_M = M_values[M_counter]
        plt.scatter(row['X'], row['Y'], color=colors[M_counter], marker='*', s=50)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means Clustering")

# Color chart
legend_elements = []
for color, M in zip(colors, M_values):
    legend_elements.append(plt.Line2D([0], [0], marker='*', color=color, label=f'M={M}', markersize=10))
plt.legend(handles=legend_elements, title='M Values', loc='center left', bbox_to_anchor=(1.05, 0.5))
plt.grid(True)

plt.show()