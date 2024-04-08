import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.svm import SVC
from joblib import dump, load
from Projet_IA import data_0
from GaussianNB_Model_Training import data_transfer, data_scaling, save_model, load_model, training_model, plot_learning_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


# KNN, Decision tree/RandomForest, SVM,, Optimisation Bayesienne to see what are the best parameters to tune our model
print("Random Forest Model")
X, y = data_transfer(data_0.filtered)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train = data_scaling(X_train)
X_test = data_scaling(X_test)

model_clf = training_model(RandomForestClassifier(), X_train, y_train, X_test, y_test)

# save_model(modele_clf, 'Vehicle_prediction_RandomForest')

# This part is to research the best parameters to maximize the model's accuracy
param_grid = {
    'n_estimators': [100, 200, 300], #100
    'criterion': ['gini', 'entropy'], #entropy
    'max_depth': [None, 10, 20], #10
    'min_samples_split': [2, 5, 10], #5
    'min_samples_leaf': [1, 2, 4], #2
    'max_features': ['sqrt', 'log2'],#log2
    'max_leaf_nodes': [None, 10, 20], # 10
    'bootstrap': [True, False] #True
}

HGSearch = HalvingGridSearchCV(model_clf, param_grid, cv=5, factor=2, max_resources=100)
HGSearch.fit(X, y)

print("Best parameters found: ", HGSearch.best_params_)
print("Best score: ", HGSearch.best_score_)


# Plot learning curve
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
plot_learning_curve(RandomForestClassifier(), 'Learning Curve For Random Forest Model', X, y, cv=5)
