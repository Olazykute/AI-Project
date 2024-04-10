import Projet_IA as P
from sklearn.naive_bayes import GaussianNB
import time

start_time = time.perf_counter()

print("GaussianNB Model")
X, y = P.data_transfer(P.data_0.filtered)

# Divide the dataset in multiple parts
X_train, X_test, y_train, y_test = P.train_test_split(X, y, test_size=0.2)

# Normalisation and standardisation
X_train, X_test = P.data_scaling(X_train, X_test)
X = P.np.concatenate((X_train, X_test), axis=0)
y = P.np.concatenate((y_train, y_test), axis=0)

# GaussianNB_model=GaussianNB()
GaussianNB_model, prediction_test, prediction_train = P.training_model(GaussianNB(), X_train, y_train, X_test, y_test)

end_time = time.perf_counter()
print("Execution time: ", (end_time - start_time)*1000)

# Print evaluation data
P.Model_Report(GaussianNB_model, X, y, y_test, prediction_test)
P.disp_confusionMatrix(GaussianNB_model, y_test,prediction_test, 'Confusion matrix for GaussianNB model')

# Save the model in a document joblib, dump to save, load is explicit
# save_model(GaussianNB_model, 'Vehicle_prediction_GaussianNB')

# No need for hyperparameters in a GaussianNB model

# Plot learning curve
P.plot_learning_curve(GaussianNB(), 'Learning Curve For GaussianNB Model', X, y, cv=5)

