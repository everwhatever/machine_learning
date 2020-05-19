import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Classified Data')
standard=StandardScaler()
standard.fit(df.drop('TARGET CLASS',axis=1))
scaled=standard.transform(df.drop('TARGET CLASS',axis=1))
df_feat=pd.DataFrame(scaled,columns=df.columns[:-1])
X=df_feat
y=df['TARGET CLASS']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(classification_report(y_test,predictions))

#
# error_rate=[]
# for i in range(1,50):
#     knn=KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     pred_i=knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))
#
# plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o')
# plt.show()
knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(classification_report(y_test,predictions))
