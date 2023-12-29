
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


dataset=pd.read_csv(r"D:\NIT\DATASCIENCE\ARNAK TASK\exam 1\exam 3\Housing.csv")

dataset.columns
dataset.isnull().sum()


dataset.drop(columns='Address', inplace=True)

X = dataset.iloc[:,:-1].values
y=dataset.iloc[:,5].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split( X,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

acc = regressor.score(X_train,y_train)
print(acc)

acc1 = regressor.score(X_test,y_test)
print(acc1)

'''  model acc-:0.918840114090963
                0.9146454505138176    '''
                
###############################################################                
from sklearn.svm import SVR
re1=SVR( degree=3,kernel="linear", gamma="auto",epsilon=0.2)
re1.fit(X, y)
re1.fit(X_train, y_train)

y_pred1=re1.predict(X_test)




acc2 = re1.score(X_train,y_train)
print(acc2)

acc3 = re1.score(X_test,y_test)
print(acc3)



'''  model acc-:0.5946757751556113
                0.5951197019320773   '''
############################################################### 



from sklearn.tree import DecisionTreeRegressor
re2=DecisionTreeRegressor( criterion='squared_error',splitter="best",random_state=0, 
                          max_depth=10,min_samples_split=7, 
                          min_samples_leaf=2)
re2.fit(X, y)

re2.fit(X_train, y_train)
y_pred3 = re2.predict(X_test) 
acc4 = re2.score(X_train,y_train)
print(acc4)

acc5 = re2.score(X_test,y_test)
print(acc5)

'''  model acc-:
0.9423157850565475
0.7711747724655933  '''

############################################################### 

from sklearn.ensemble import RandomForestRegressor
re3=RandomForestRegressor(random_state=0,n_estimators=100,criterion="squared_error",
                          min_samples_split=4, max_depth=9,min_samples_leaf=1)
re3.fit(X, y)


re3.fit(X_train, y_train)
y_pred4 = re3.predict(X_test)
acc6 = re3.score(X_train,y_train)
print(acc6)

acc7 = re3.score(X_test,y_test)
print(acc7)
############################################################### 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((5000,1)).astype(int), values = X, axis = 1) 



import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
