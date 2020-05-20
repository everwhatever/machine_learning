import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

cancer=load_breast_cancer()
#print(cancer.keys())
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#print(df.head())
X=df
y=cancer['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

model=SVC()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


"""
#looking for best best_params_

param_grid={'C':[x for x in range(1,10)],'gamma':[ 0.001,0.002,0.003,0.004,0.005,0.006]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)

grid_pred=grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
"""
