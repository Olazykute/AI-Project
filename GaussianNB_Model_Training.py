from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import ValidationCurveDisplay, cross_val_score
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import Projet_IA as P

# data=pl.read_csv("Features_by_window_size/sero_features_4.csv")
# print (data)

def data_transfer(df):
    X = df[:, 1:len(df)] #les caractéristiques
    y = df[:, 0]  #les résulats (classes)
    X = X.to_numpy()
    y = y.to_numpy()
    return X, y

def data_scaling(x):
    scaler = StandardScaler()
    x=scaler.fit_transform(x)  # Fit the scaler to your data and transform the matrix
    return x

def save_model(model, nom):
    dump( model, nom+'.joblib')
    return

def load_model(nom):
    model=load(nom)
    return model

def training_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train) # Model fit c'est l'entrainement du modèle

    y_pred = model.predict(X_test) # Prediction sur l'ensemble de test
    y_pred2 = model.predict(X_train) # Prédiction sur l'ensemble d'entrainement
    print("precsion en test: ", accuracy_score(y_test, y_pred))
    print("precsion en entrainement: ", accuracy_score(y_train, y_pred2))

    report = classification_report(y_test, y_pred)
    print(print("Classification Report on test:\n", report))
    # Le classification report donne la précision du modèle par classe

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=2)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    # La validation croisée teste le modèle 'cv' fois de suite dans l'ensemble de test.

    return model

def disp_confusionMatrix():
    
    return

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
GaussianNB_model=training_model(GaussianNB(), X_train, y_train, X_test, y_test)

# enregistre le modèle dans un fichier joblib, dump pour enregistrer, load pour le charger
# save_model(GaussianNB_model, 'Vehicle_prediction_GaussianNB')

# No need for hyperparameters in a GaussianNB model