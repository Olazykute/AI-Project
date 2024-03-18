import numpy as np
import matplotlib.pyplot as plt
import polars as pl

data_0 = pl.read_csv("Features_by_window_size/sero_features_4.csv")
print(data_0)

def show_corr(df, threshold=0.7): # Show highly correlated columns
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Extract highly correlated columns
    highly_correlated_cols = []

    for i, col in enumerate(df.columns):
        for j in range(i + 1, len(df.columns)):
            corr_value = corr_matrix[(i, j)]
            if abs(corr_value) >= threshold:
                highly_correlated_cols.append((df.columns[i], df.columns[j]))

    print("Highly correlated columns:")
    for col_pair in highly_correlated_cols:
        print(col_pair)
    return highly_correlated_cols
    
def drop_corr(df , highly_correlated_cols): # Drop highly correlated columns
    for i, col_pair in enumerate(highly_correlated_cols):
        i_col, j_col = col_pair
        if j_col not in [col[0] for col in highly_correlated_cols[:i]]:
            df = df.drop(j_col)
    return df
                

def plot_acc_gyro(df): # Plot acc and gyro mean
    x = np.linspace(0, df.len(), df.len())
    y = df[:,"AccMeanX"]
    y2 = df[:,"AccMeanY"]
    y3 = df[:,"AccMeanZ"]

    y4 = df[:,"GyroMeanX"]
    y5 = df[:,"GyroMeanY"]
    y6 = df[:,"GyroMeanZ"]

    plt.figure()
    plt.plot(x, y, color="red")
    plt.plot(x, y2, color="blue")
    plt.plot(x, y3, color="green")

    plt.figure()
    plt.plot(x, y4, color="red")
    plt.plot(x, y5, color="blue")
    plt.plot(x, y6, color="green")
    plt.show()
    
def correlation_matrix(df): # plot correlation matrix
    
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm', interpolation='nearest')
    fig.colorbar(cax)
    plt.colorbar(label='Corrélation')
    plt.title("Matrice de corrélation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    
hc_cols = show_corr(data_0)
data_0_filtered = drop_corr(data_0, hc_cols)
show_corr(data_0)
