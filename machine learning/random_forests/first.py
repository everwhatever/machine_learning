import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('kyphosis.csv')
#print(df.head())
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
rfc=RandomForestClassifier(n_estimators=48)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
