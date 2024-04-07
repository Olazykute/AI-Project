import numpy as np
import matplotlib.pyplot as plt
import polars as pl


class Data:
    def __init__(self, path):
        self.raw = pl.read_csv(path)
        self.filtered = self.raw


data_0 = Data("Features_by_window_size/sero_features_4.csv")

def cal_corr(df, threshold=0.7):  # Calculate and show highly correlated columns
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


def drop_corr(df, highly_correlated_cols):  # Drop highly correlated columns
    for i, col_pair in enumerate(highly_correlated_cols):
        i_col, j_col = col_pair
        if j_col not in [col[0] for col in highly_correlated_cols[:i]]:
            df = df.drop(j_col)
    return df


def plot_acc_gyro(df):  # Plot acc and gyro mean
    x = np.linspace(0, df.len(), df.len())
    y = df[:, "AccMeanX"]
    y2 = df[:, "AccMeanY"]
    y3 = df[:, "AccMeanZ"]

    y4 = df[:, "GyroMeanX"]
    y5 = df[:, "GyroMeanY"]
    y6 = df[:, "GyroMeanZ"]

    plt.figure()
    plt.plot(x, y, color="red")
    plt.plot(x, y2, color="blue")
    plt.plot(x, y3, color="green")

    plt.figure()
    plt.plot(x, y4, color="red")
    plt.plot(x, y5, color="blue")
    plt.plot(x, y6, color="green")
    plt.show()


def correlation_matrix(df):  # plot correlation matrix

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm', interpolation='nearest')
    fig.colorbar(cax)
    plt.title("Matrice de corrélation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


def normalize(df: pl.DataFrame):
    for col in df.columns:
        if col == 'Target':
            continue
        min_val = df[col].min()
        max_val = df[col].max()
        normalized_col = (df[col] - min_val) / (max_val - min_val)
        df = df.drop(col).with_columns([normalized_col.rename(col)])
    return df


print(data_0.raw)
hc_cols = cal_corr(data_0.raw)
data_0.filtered = drop_corr(data_0.raw, hc_cols)
#cal_corr(data_0.filtered)
#correlation_matrix(data_0.filtered)

'''
# To observe the filtered data
print(data_0.filtered)
data_0.filtered.write_csv('Sensor filtered.csv')
'''
