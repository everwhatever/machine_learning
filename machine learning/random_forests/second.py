import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('loan_data.csv')
# print(df.head())
# print(df['purpose'].value_counts())

purpose=pd.get_dummies(df['purpose'],drop_first=True)
df=pd.concat([df,purpose],axis=1)
df.drop('purpose',axis=1,inplace=True)
#print(df)

X=df.drop('not.fully.paid',axis=1)
y=df['not.fully.paid']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

rfc=RandomForestClassifier(n_estimators=300)

rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
