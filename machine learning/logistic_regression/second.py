import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data=pd.read_csv('advertising.csv')

data.drop(['Ad Topic Line', 'City', 'Timestamp','Country'],axis=1,inplace=True)
# country=pd.get_dummies(data['Country'],drop_first=True)
# data=pd.concat([data,country])
X=data.drop('Clicked on Ad',axis=1)
y=data['Clicked on Ad']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

print(classification_report(y_test,predictions))

# print(data.head())

# sns.heatmap(data.isnull())

plt.show()
