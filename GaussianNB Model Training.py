import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ValidationCurveDisplay
from joblib import dump, load
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler
import Projet_IA as P

# data=pl.read_csv("Features_by_window_size/sero_features_4.csv")
# print (data)

X = P.data_0.filtered[:, 1:len(P.data_0.filtered)] #les caractéristiques
y = P.data_0.filtered[:, 0]  #les résulats (classes)

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train)
# Normalisation et standardisation
scaler = StandardScaler()
# Fit the scaler to your data and transform the matrix
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print(X_train)

modele = GaussianNB()
modele.fit(X_train, y_train)

y_pred = modele.predict(X_test)
y_pred2 = modele.predict(X_train)
print("precsion en test: ", accuracy_score(y_test, y_pred))
print("precsion en entrainement: ", accuracy_score(y_train, y_pred2))

# Inutile pour un GaussienNB
#ValidationCurveDisplay.from_estimator(
#   GaussianNB(kernel="linear"), X, y, param_name="", param_range=np.logspace(-7, 3, 10)
#)
plt.show()

# ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# enregistre le modèle dans un fichier joblib, dump pour enregistrer, load pour le charger
dump( modele, 'Vehicle_prediction_GaussianNB.joblib')

# No need for hyperparameters in a GaussianNB model