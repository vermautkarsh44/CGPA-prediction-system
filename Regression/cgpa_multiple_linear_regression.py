#Regression
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv("Student's Information_cse.csv")
X= dataset.iloc[:,:12].values
Y=dataset.iloc[:,12].values

#label encoder
#one hot encoder
from sklearn.preprocessing import LabelEncoder
#from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
#ct=ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[3])],remainder='passthrough')
#x=ct.fit_transform(x)

#missing values
from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan,strategy="mean")
simpleimputer=simpleimputer.fit(X[:,9:11])
X[:,9:11]=simpleimputer.transform(X[:,9:11])

#spliting the data set into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn import metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred=regressor.predict(X_test)

#Visualizing the results
plt.title("Multiple Linear Regression")
plt.plot(y_test)
plt.plot(y_pred)
plt.xlabel("Number of students")
plt.ylabel("SGPA 4")
plt.legend(["Test Values","Predicted Values"])
plt.show()

#printing training and testing scores
print('Train Score: ', regressor.score(X_train, y_train))  
print('Test Score: ', regressor.score(X_test, y_test))  



