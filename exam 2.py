

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset=pd.read_csv(r"D:\NIT\DATASCIENCE\ARNAK TASK\exam 1\exam 2\health cost.csv")
dataset.isnull().sum()

X = dataset.iloc[:,:-1].values
y=dataset.iloc[:,6].values

#X=pd.get_dummies(X,dtype=int)

from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()



labelencoder_X=labelencoder_X.fit(X[:,1])
(X[:,1])=labelencoder_X.transform(X[:,1])

labelencoder_X=labelencoder_X.fit(X[:,4])
(X[:,4])=labelencoder_X.transform(X[:,4])

labelencoder_X=labelencoder_X.fit(X[:,5])
(X[:,5])=labelencoder_X.transform(X[:,5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split( X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.neighbors import KNeighborsRegressor
re9=KNeighborsRegressor(n_neighbors=6, weights="distance",algorithm="kd_tree")
re9.fit(X, y)


# predicto 
re9.fit(X_train, y_train)

acc8 = re9.score(X_train,y_train)
print(acc8)

acc9 = re9.score(X_test,y_test)
print(acc9)
y_pred1 = re9.predict(X_test)




from sklearn.svm import SVR
re6=SVR( degree=5,kernel="poly")
re6.fit(X, y)

# predicto 
re6.fit(X_train, y_train)
y_pred2 = re6.predict(X_test)

acc6 = re6.score(X_train,y_train)
print(acc6)

acc7 = re6.score(X_test,y_test)
print(acc7)




from sklearn.tree import DecisionTreeRegressor
re2=DecisionTreeRegressor( criterion='absolute_error',splitter="random",random_state=0, 
                          max_depth=10,min_samples_split=8)
re2.fit(X, y)

re2.fit(X_train, y_train)
y_pred3 = re2.predict(X_test) 
acc4 = re2.score(X_train,y_train)
print(acc4)

acc5 = re2.score(X_test,y_test)
print(acc5)


from sklearn.ensemble import RandomForestRegressor
re7=RandomForestRegressor(random_state=0,n_estimators=100,criterion="friedman_mse",
                          min_samples_split=6, max_depth=9,min_samples_leaf=1)
re7.fit(X, y)


re7.fit(X_train, y_train)
y_pred4 = re7.predict(X_test)
acc = re7.score(X_train,y_train)
print(acc)

acc1 = re7.score(X_test,y_test)
print(acc1)


X_train.shape, y_train.shape




''''

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1338,1)).astype(int), values = X, axis = 1) 


import statsmodels.api as sm
X_opt = X[:,[0,1,3,4,5,6]].astype(int)


regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()




accuracy = mean_squared_error(y_test, y_pred4)

acc = re7.score(X_train,y_train)
print(acc)
acc1 = re7.score(X_test,y_test)
print(acc1)
""""

