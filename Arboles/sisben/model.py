
# -*- coding: utf-8 -*-
"""
Created on Fri Ago 14 2020
@author: manish
"""
# General
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
# Keras

# sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import StackingClassifier
# Reproducibility
# seed = 11
# from tensorflow import set_random_seed
# set_random_seed(seed)


# print("\033[91m Lectura de datos \033[0m")

# #************ Preparing the data ************#
def get_dataset():
	# Read the data 
	data = pd.read_csv("base_calculos.csv")
	df = pd.DataFrame(data = data)
	# df = df.dropna()

	# # Choose a subset of the complete data 
	df_sample = df.sample(frac = 0.1,random_state = 0)
	
	independientes=df_sample[["zona", "discapa", "nivel", "edad", "teneviv", "pared",
							"piso", "energia", "energia", "alcanta","gas", "telefono", 
							"basura", "acueduc", "elimbasura", "sanitar","ducha", 
							"llega","cocinan", "preparan", "alumbra" , "usotele", 
							"nevera", "lavadora", "tvcolor", "tvcable","calenta", 
							"horno", "aire", "computador" , "equipo", "moto", "tractor",
							 "auto1" , "bieraices", "area_total"]]

# independientes=pd.get_dummies(independientes, columns = independientes.columns , dtype=float)

# # Eliminacion con baja varianza 
# x_train = independientes.to_numpy()
# y_train = df_sample["subsidiado"].to_numpy()

# x_train = SelectKBest(chi2, k=95).fit_transform(x_train, y_train)
# X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train,test_size = 0.2,random_state =0)

# print("\033[91m Construct the models \033[0m")
# LR = LogisticRegression(random_state = 0, solver = "saga")
# # LR.fit(X_train,Y_train)
# # pred = LR.predict(X_test)
# # print("##### Logistic Regression  results #####")
# # print('Precision validacion        = {:.2f}'.format(LR.score(X_test, Y_test)))
# # print(classification_report(Y_test, pred))


# ### Decision Tree
# dt = DecisionTreeClassifier(criterion="entropy", random_state = 1,splitter = "best")
# # clf_dt = dt.fit(X_train,Y_train)
# # dot_data = tree.export_graphviz(clf_dt,
# #                                # feature_names = data.columns[:-4],
# #                                filled=True,
# #                                rounded = True,
# #                                special_characters=True,
# #                                out_file="tree.dot")
# # pred = dt.predict(X_test)
# # print("##### Decision tree results #####")
# # print('Precision validacion        = {:.2f}'.format(dt.score(X_test, Y_test)))
# # print(classification_report(Y_test,pred))

# # Defin a list of models to evaluate
# def get_models():
# 	models = dict()
# 	models["LR"] = LogisticRegression(random_state = 0, solver = "saga")
# 	models["KNN"] = KNeighborsClassifier()
# 	models["CART"] = DecisionTreeClassifier()
# 	models["SVM"] = SVM()
# 	models["BAYES"] = GaussianNB()
# 	return models

# # classifiers = [('Logistic Regression', LR),('Classification Tree', dt)]

# # print("\033[91m running models \033[0m")

# # for clf_name, clf in classifiers:    
# #     clf.fit(X_train, Y_train)    
# #     y_pred = clf.predict(X_test)
# #     accuracy = accuracy_score(Y_test, y_pred)
# #     print('{:s} : {:.3f}'.format(clf_name, accuracy))
# #     print(classification_report(Y_test,y_pred))



# # print("\033[91m running metamodel \033[0m")

# # vc = VotingClassifier(estimators=classifiers)     
# # vc.fit(X_train, Y_train)   
# # y_pred = vc.predict(X_test)
# # accuracy = accuracy_score(Y_test, y_pred)
# # print('Voting Classifier: {:.3f}'.format(accuracy))
# # print(classification_report(Y_test,y_pred))

