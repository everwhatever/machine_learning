import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df=pd.read_csv('titanic_train.csv')

def input_age(data):
    pclass=data[0]
    age=data[1]
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 30
        else:
            return 25
    else:
        return age

df['Age']=df[['Pclass','Age']].apply(input_age,axis=1)
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)

Sex=pd.get_dummies(df['Sex'],drop_first=True)
Embarked=pd.get_dummies(df['Embarked'],drop_first=True)
df=pd.concat([df,Sex,Embarked],axis=1)

df.drop('Sex',axis=1,inplace=True)
df.drop('Embarked',axis=1,inplace=True)

# print(df.head())
# #print(df.groupby(['Pclass']).mean())
# sns.heatmap(df.isnull())


y=df['Survived']
X=df.drop('Survived',axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
print(classification_report(y_test,predictions))

plt.show()
