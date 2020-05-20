import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

iris = sns.load_dataset('iris')
#print(iris.head())
#print(iris['species'].value_counts())

X=iris.drop('species',axis=1)
y=iris['species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

model=SVC()
model.fit(X_train,y_train)
predictions=model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#print(sns.pairplot(iris))


"""
#looking for best_params_

param_grid={'C':[1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)

grid_pred=grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
"""
plt.show()
