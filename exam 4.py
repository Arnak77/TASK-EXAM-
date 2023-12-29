
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"D:\NIT\DATASCIENCE\ARNAK TASK\exam 1\exam 4\data.csv")

dataset.columns
dataset.isnull().sum()


X = dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()


labelencoder_X=labelencoder_X.fit(X[:,3])
(X[:,3])=labelencoder_X.transform(X[:,3])


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
''''
0.9500009880362248
0.9386861070938135'''

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
'''
0.9163713960635558
0.8742484023135983
'''
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
'''0.9787022018638384
0.9351030349939689'''


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

'''0.9807506027749443
0.9647462006086848'''
############################################################### 


''''
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 



import statsmodels.api as sm
X_opt = X[:,[0,1]].astype(float)
y_numeric = np.asarray(y).astype(float)




regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()




'''