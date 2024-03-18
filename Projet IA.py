import numpy as np
import matplotlib.pyplot as plt
import polars as pl

data_0 = pl.read_csv("Features_by_window_size/sero_features_4.csv")
print(data_0)

""" 
plt.hist(data_0[:,"Target"],bins=10)
plt.xlabel("Target")
plt.show()
"""

# Calculate correlation matrix
corr_matrix = data_0.corr()

# Define correlation threshold
threshold = 0.7

# Extract highly correlated columns
highly_correlated_cols = []

for i, col in enumerate(data_0.columns):
    for j in range(i + 1, len(data_0.columns)):
        corr_value = corr_matrix[(i, j)]
        if abs(corr_value) >= threshold:
            highly_correlated_cols.append((data_0.columns[i], data_0.columns[j]))

print("Highly correlated columns:")
for col_pair in highly_correlated_cols:
    print(col_pair)
    

target = data_0[:,"Target"]
x = np.linspace(0,target.len(),target.len())
y = data_0[:,"AccMeanX"]
y2 = data_0[:,"AccMeanY"]
y3 = data_0[:,"AccMeanZ"]

y4 = data_0[:,"GyroMeanX"]
y5 = data_0[:,"GyroMeanY"]
y6 = data_0[:,"GyroMeanZ"]


plt.figure()
plt.plot(x,y,color="red")
plt.plot(x,y2,color="blue")
plt.plot(x,y3,color="green")

plt.figure()
plt.plot(x,y4,color="red")
plt.plot(x,y5,color="blue")
plt.plot(x,y6,color="green")

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Corrélation')
plt.title("Matrice de corrélation")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.show()

