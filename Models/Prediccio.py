from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle 

#carregam el dataset 
iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

# dividim la mostra en dues parts Un 70% per a l'entrenament i Un 30% per a l'avaluació

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1, stratify = y)

# normalitzar amb un escalat estàndard,
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# cream instacies de cada model
Regresio = LogisticRegression()
Svm = SVC()
Arbre = DecisionTreeClassifier()
Knn = KNeighborsClassifier()

# Entrenam els 4 models
Regresio.fit(X_train_std, y_train)
Svm.fit(X_train_std, y_train)
Arbre.fit(X_train_std, y_train)
Knn.fit(X_train_std, y_train)

# Realizam prediccions per cada model 
y_pred_logreg = Regresio.predict(X_test_std)
y_pred_svm = Svm.predict(X_test_std)
y_pred_tree = Arbre.predict(X_test_std)
y_pred_knn = Knn.predict(X_test_std)

# Serialitzar i guardar els  models

with open('Regresio_model.pkl', 'wb') as file:
    pickle.dump(Regresio, file)
with open('Svm_model.pkl', 'wb') as file:
    pickle.dump(Svm, file)
with open('Arbre_model.pkl', 'wb') as file:
    pickle.dump(Arbre, file)
with open('Knn_model.pkl', 'wb') as file:
    pickle.dump(Knn, file)

# Guardar Scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)