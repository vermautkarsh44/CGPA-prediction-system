#Naive Bayes Classification

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data set
dataset = pd.read_csv("classification data.csv")
x=dataset.iloc[:,0:12].values
y=dataset.iloc[:,12].values

#label encoder
#one hot encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
#ct=ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[3])],remainder='passthrough')
#x=ct.fit_transform(x)

#spliting the data set into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the test set results
y_pred=classifier.predict(X_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Error calculation
from sklearn import metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

print_error(X_test,y_test, classifier)

print('Train Score: ', classifier.score(X_train, y_train))  
print('Test Score: ', classifier.score(X_test, y_test)) 

from sklearn.metrics import r2_score
r2_test = r2_score(y_test, y_pred)
print('R2 score: ',r2_test)

#Graph
plt.title("Naive Bayes Classifier")
plt.plot(y_test)
plt.plot(y_pred)
plt.xlabel("Number of students")
plt.ylabel("SGPA 4")
plt.legend(["Test Values","Predicted Values"])
plt.show()

