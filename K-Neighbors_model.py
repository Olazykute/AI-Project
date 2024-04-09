from sklearn.neighbors import KNeighborsClassifier
import Projet_IA as P
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import time

start_time = time.perf_counter()

X, y = P.data_transfer(P.data_0.filtered)

# Divide the dataset in multiple parts
X_train, X_test, y_train, y_test = P.train_test_split(X, y, test_size=0.2)

# Normalisation and standardisation
X_train, X_test = P.data_scaling(X_train, X_test)
X = P.np.concatenate((X_train, X_test), axis=0)
y = P.np.concatenate((y_train, y_test), axis=0)

P.np.random.seed(42)

KNeighbors_model, prediction_test, prediction_train = P.training_model(KNeighborsClassifier(n_neighbors=4), X_train, y_train, X_test, y_test)

end_time = time.perf_counter()
print("Execution time: ", (end_time - start_time)*1000)

P.Model_Report(KNeighbors_model, X, y, y_test, prediction_test)
P.disp_confusionMatrix(KNeighbors_model, y_test,prediction_test, 'Confusion matrix for GaussianNB model')

# Save the model in a document joblib, dump to save, load is explicit
# save_model(KNeighbors_model_model, 'Vehicle_prediction_KNeighbors_model')

KNeighbors_model.kneighbors_graph()

# Plot learning curve
P.plot_learning_curve(KNeighbors_model, 'Learning Curve For Kneighbors_Model', X, y, cv=5)
