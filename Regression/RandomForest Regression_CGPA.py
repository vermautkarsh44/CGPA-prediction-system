#Random forest Regression

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data set
dataset = pd.read_csv("Student's Information_cse.csv")
x=dataset.iloc[:,:12].values
y=dataset.iloc[:,12].values
 
#label encoder
#one hot encoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
#ct=ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[3])],remainder='passthrough')
#x=ct.fit_transform(x)

#missing values
from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan,strategy="mean")
simpleimputer=simpleimputer.fit(x[:,9:11])
x[:,9:11]=simpleimputer.transform(x[:,9:11])

#spliting the data set into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting the Random forest Regression model to the data set
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=200,random_state=0)
regressor.fit(x,y)

#Predict Result
y_pred=regressor.predict(X_test)

#Visualizing the results
plt.title("Random Forest Regression")
plt.plot(y_test)
plt.plot(y_pred)
plt.xlabel("Number of students")
plt.ylabel("SGPA 4")
plt.legend(["Test Values","Predicted Values"])
plt.show()

#Error calculation
from sklearn import metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

print_error(X_test,y_test, regressor)

print('Train Score: ', regressor.score(X_train, y_train))  
print('Test Score: ', regressor.score(X_test, y_test)) 

from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred)
print('R2 score: ',r2_test)

