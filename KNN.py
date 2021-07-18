# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:07:25 2021

@author: tales
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

url = "Tabelas - KNN/Compiled F Filter.csv"

dataset = pd.read_csv(url, delimiter=';')

print(dataset.head())

X = dataset.iloc[:, :-1].values

print(X)

y = dataset.iloc[:, 22].values

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=1)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




