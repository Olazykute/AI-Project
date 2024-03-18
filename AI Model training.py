import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.svm import SVC
from joblib import dump, load
 

data=pl.read_csv("Features_by_window_size/sero_features_4.csv")
print (data)

X = data[:, 1:len(data)] #les caractéristiques
y = data[:, 0]  #les résulats (classes)

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modele = GaussianNB()
modele.fit(X_train, y_train)

y_pred = modele.predict(X_test)

print("precsion : ", accuracy_score(y_test, y_pred))

ValidationCurveDisplay.from_estimator(
   SVC(kernel="linear"), X, y, param_name="C", param_range=np.logspace(-7, 3, 10)
)
plt.show()

# enregistre le modèle dans un fichier joblib, dump pour enregistrer, load pour le charger
# dump( modele, 'Vehicle_prediction_GaussianNB.joblib')
