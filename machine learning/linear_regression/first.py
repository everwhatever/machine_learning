import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv('USA_Housing.csv')
print(df.columns)
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)

# cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
# print(cdf)
predictions=lm.predict(X_test)
#print(predictions)
plt.scatter(y_test,predictions)

print(f'mean_absolute_error: {metrics.mean_absolute_error(y_test,predictions)}')
print(f'mean_squared_error: {metrics.mean_squared_error(y_test,predictions)}')
print(f'mean_root_squared_error: {np.sqrt(metrics.mean_squared_error(y_test,predictions))}')



plt.show()

# df=pd.read_csv('USA_Housing.csv')
# X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#         'Avg. Area Number of Bedrooms', 'Area Population']]
# y=df['Price']
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
#
# lm=LinearRegression()
# lm.fit(X_train,y_train)
# predictions=lm.predict(X_test)
# print(metrics.mean_absolute_error(y_test,predictions))
# print(metrics.mean_squared_error(y_test,predictions))
# print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
