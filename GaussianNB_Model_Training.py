from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import ValidationCurveDisplay, cross_val_score, learning_curve
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import Projet_IA as P
import matplotlib.pyplot as plt
import numpy as np


# data=pl.read_csv("Features_by_window_size/sero_features_4.csv")
# print (data)


def data_transfer(df):
    X = df[:, 1:len(df)]  # les caractéristiques
    y = df[:, 0]  # les résulats (classes)
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
    model.fit(X_train, y_train)  # Model fit c'est l'entrainement du modèle

    y_pred = model.predict(X_test)  # Prediction sur l'ensemble de test
    y_pred_train = model.predict(X_train)# Prédiction sur l'ensemble d'entrainement
    print("precsion en test: ", accuracy_score(y_test, y_pred))
    print("precsion en entrainement: ", accuracy_score(y_train, y_pred_train))

    report = classification_report(y_test, y_pred)
    print(print("Classification Report on test:\n", report))
    # The classification report gives the precision, recall, f1-score and support for each class.

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=3)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    # The cross validation score tests the model'cv' times in the test set. 

    disp_confusionMatrix(model, y_test, y_pred)

    return model


def disp_confusionMatrix(modele, test, prediction):
    cm=confusion_matrix(test, prediction, labels=modele.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modele.classes_)
    disp.plot()
    P.plt.show()
    return

print("GaussianNB Model")
X, y = data_transfer(P.data_0.filtered)

# Division du dataset en différentes parties
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("Pre-scaling")
print(X_train)

# Normalisation et standardisation
X_train = data_scaling(X_train)
X_test = data_scaling(X_test)
print("Post-scaling")
print(X_train)

# GaussianNB_model=GaussianNB()
GaussianNB_model = training_model(
    GaussianNB(), X_train, y_train, X_test, y_test)

# enregistre le modèle dans un fichier joblib, dump pour enregistrer, load pour le charger
# save_model(GaussianNB_model, 'Vehicle_prediction_GaussianNB')

# No need for hyperparameters in a GaussianNB model

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 100)):
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

'''
# Plot learning curve
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
plot_learning_curve(GaussianNB(), 'Learning Curve For GaussianNB Model', X, y, cv=5)
'''