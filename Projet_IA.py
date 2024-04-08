import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import ValidationCurveDisplay, cross_val_score, learning_curve
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


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
        j_col = col_pair
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


def data_transfer(df):
    X = df[:, 1:len(df)]  # Caracteristics / Parameters
    y = df[:, 0]  # Results (classes) / target
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y


def data_scaling(x):
    scaler = StandardScaler()
    # Fit the scaler to your data and transform the matrix
    x = scaler.fit_transform(x)
    return x


def save_model(model, nom):
    dump(model, nom+'.joblib')
    return


def load_model(nom):
    model = load(nom)
    return model


def training_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # Model fit is the training of the model

    y_pred = model.predict(X_test)  # Prediction on the test dataset
    y_pred_train = model.predict(X_train)  # Prediction on the training dataset
    print("precsion en test: ", accuracy_score(y_test, y_pred))
    print("precsion en entrainement: ", accuracy_score(y_train, y_pred_train))

    return model, y_pred, y_pred_train


def Model_Report(model, X, y , y_test, y_pred):

    report = classification_report(y_test, y_pred)
    print(print("Classification Report on test:\n", report))
    # The classification report gives the precision, recall, f1-score and support for each class.

    # Perform cross-validation on the entire dataset
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    # The cross validation score tests the model'cv' times in the test set.


def disp_confusionMatrix(modele, test, prediction, titre):
    cm = confusion_matrix(test, prediction, labels=modele.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=modele.classes_)
    disp.plot()
    plt.title(titre)
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 50)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

def plot_gauss(data_gauss):
    plt.figure(figsize=(12, 12))

    i=0
    for col in data_gauss.columns:
        # Calculer la moyenne et l'écart type
        moy = data_gauss[col].mean()
        std = data_gauss[col].std()

        # Création de la courbe de répartition gaussienne
        x = np.linspace(moy - 3*std, moy + 3*std, 100)
        y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - moy) / std) ** 2)

        # Plot de la courbe de répartition gaussienne
        plt.subplot(4, 4, i + 1)
        i += 1
        plt.plot(x, y, color='red')
        plt.hist(data_gauss[col], bins=50, density=True, color='blue')
        #plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')
    plt.legend()
    plt.tight_layout()        
    plt.show()

# print(data_0.raw)
hc_cols = cal_corr(data_0.raw)
data_0.filtered = drop_corr(data_0.raw, hc_cols)
# cal_corr(data_0.filtered)
# correlation_matrix(data_0.filtered)

data_gauss = data_0.filtered.drop(columns='Target')
plot_gauss(data_gauss)


'''
# To observe the filtered data
print(data_0.filtered)
data_0.filtered.write_csv('Sensor filtered.csv')
'''
