import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv('Ecommerce Customers')
#sns.jointplot(df['Time on Website'],df['Yearly Amount Spent'])
X=df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=df[['Yearly Amount Spent']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
print(f'mean_absolute_error: {metrics.mean_absolute_error(y_test,predictions)}')
print(f'mean_squared_error: {metrics.mean_squared_error(y_test,predictions)}')
print(f'mean_root_squared_error: {np.sqrt(metrics.mean_squared_error(y_test,predictions))}')
plt.show()
