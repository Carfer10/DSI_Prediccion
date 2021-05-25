# -*- coding: utf-8 -*-
"""
@author: Fernando Vallejo Banegas
@author: Luis Jiménez Navajas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leemos los datos train, test y validacion
dataset_train = pd.read_csv('../Data/train.csv')
dataset_test = pd.read_csv('../Data/test.csv')
dataset_valid = pd.read_csv('../Data/valid.csv')

# Separamos X de Y
Y_train_df = np.asarray(dataset_train["Resultado"])
Y_test_df = np.asarray(dataset_test["Resultado"])

X_train_df = dataset_train.drop(['Equipo1','Equipo2','Resultado'],axis=1)
X_test_df = dataset_test.drop(['Equipo1','Equipo2','Resultado'],axis=1)
X_valid_df = dataset_valid.drop(['Equipo1','Equipo2'],axis=1)

# Calculamos la correlación entre las variables
corr = X_train_df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(X_train_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(X_train_df.columns)
ax.set_yticklabels(X_train_df.columns)
plt.show()

# Normalizamos los datos
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
X_train_norm = min_max_scaler.fit_transform(X_train_df)
X_test_norm = min_max_scaler.fit_transform(X_test_df)
X_valid_norm = min_max_scaler.fit_transform(X_valid_df)

# Calculamos PCA y su EVR
from sklearn.decomposition import PCA
estimator = PCA (n_components = 2)
X_train_pca = estimator.fit_transform(X_train_norm)
print(estimator.explained_variance_ratio_)
X_test_pca = estimator.fit_transform(X_test_norm)
X_valid_pca = estimator.fit_transform(X_valid_norm)

plt.scatter(X_train_pca[:,0], X_train_pca[:,1] ,s=25)

for i in range(0,len(X_train_pca)):
    plt.annotate(f"{dataset_train['Equipo1'][i]}-{dataset_train['Equipo2'][i]}",
                 (X_train_pca[i,0], X_train_pca[i,1]))
    
plt.grid()
plt.show()

# Preparamos los datos para la clasificacion
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_train_norm, Y_train_df, test_size=0.40, random_state=42)
X_train = X_train_norm
y_train = Y_train_df
X_test = X_test_norm
y_test = Y_test_df

from sklearn.metrics import classification_report

# Lanzamos un Perceptrón Multicapa
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                    hidden_layer_sizes=(10, 3), random_state=1)

clf.fit(X_train, y_train)

# Calculamos matriz de confusión
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, X_test, y_test)  
plt.title("Multi Layer Perceptron")
plt.show()
resultados = clf.predict(X_valid_norm)
print("Multi Layer Perceptron")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Lanzamos Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
plot_confusion_matrix(gnb, X_test, y_test)
plt.title("Gaussian Naive Bayes")
plt.show()
resultados = gnb.predict(X_valid_norm)
print("Gaussian Naive Bayes")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred))

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
plot_confusion_matrix(mnb, X_test, y_test)  
plt.title("Multinomial Naive Bayes")
plt.show()
resultados = mnb.predict(X_valid_norm)
print("Multinomial Naive Bayes")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = mnb.predict(X_test)
print(classification_report(y_test, y_pred))


bnb = BernoulliNB()
bnb.fit(X_train,y_train)
plot_confusion_matrix(bnb, X_test, y_test)  
plt.title("Bernoulli Naive Bayes")
plt.show()
resultados = bnb.predict(X_valid_norm)
print("Bernoulli Naive Bayes")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = bnb.predict(X_test)
print(classification_report(y_test, y_pred))

# Ejecutamos KNN
from sklearn import neighbors

n_neighbors = 3

for weights in ['uniform', 'distance']:
    knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    
    plot_confusion_matrix(knn, X_test, y_test)  
    plt.title(f"KNN - {weights}")
    plt.show()
    resultados = knn.predict(X_valid_norm)
    print(f"KNN - {weights}")
    for i in range(len(dataset_valid)):
        print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))


# Regresión Logística
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

plot_confusion_matrix(lr, X_test, y_test)  
plt.title("Logistic Regression")
plt.show()
resultados = lr.predict(X_valid_norm)
print("Logistic Regression")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))


# Árboles de decisión
from sklearn import tree
tree1 = tree.DecisionTreeClassifier()
tree1.fit(X_train, y_train) 

plot_confusion_matrix(tree1, X_test, y_test)  
plt.title("Base Decision Tree")
plt.show()
resultados = tree1.predict(X_valid_norm)
print("Base Decision Tree")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = tree1.predict(X_test)
print(classification_report(y_test, y_pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = 2) # parameters (n_estimators)
rf.fit(X_train, y_train)

plot_confusion_matrix(rf, X_test, y_test)  
plt.title("Random Forest")
plt.show()
resultados = rf.predict(X_valid_norm)
print("Random Forest")
for i in range(len(dataset_valid)):
    print(f"{dataset_valid['Equipo1'][i]}-{dataset_valid['Equipo2'][i]}\t{resultados[i]}")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

print(pd.DataFrame({'Attributes':  X_train_df.columns, 'Decision Tree': tree1.feature_importances_, 'Random Forests':rf.feature_importances_}))