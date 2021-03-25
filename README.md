# DATA 310 MIDTERM PROJECT

## 1. How many observations?

from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/weatherHistory.csv')

df.shape = (96453, 12)

## 2. How many features are nominal variables?

df.head(2) = (Formatted Date Summary Precip Type Temperature (C) Apparent Temperature (C) Humidity Wind Speed (km/h) Wind Bearing (degrees) Visibility (km) Loud Cover Pressure (millibars) Daily Summary) = 3 nominal variables (Summary, Precip Type, Daily Summary)

## 3. Use Temperature to Predict Humidity

x = np.array(df['Temperature (C)'])

y = np.array(df['Humidity'])

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as MSE

lm = LinearRegression()

model = lm.fit(x.reshape(-1,1),y.reshape(-1,1))

yhat = model.predict(x.reshape(-1,1))

np.sqrt(MSE(y,yhat)) = 0.1514437964005473

## 4. If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state=2020, the Ridge model with alpha =0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):

from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=2020) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain] Xtest = X[idxtest] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() xtrainscaled = scale.fit_transform(Xtrain) xtestscaled = scale.fit_transform(Xtest) model.fit(xtrainscaled,ytrain) yhat = model.predict(xtestscaled) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

model = Ridge(alpha = 0.1)

x = np.array(df['Temperature (C)'])

y = np.array(df['Humidity'])

DoKFold(x.reshape(-1,1),y.reshape(-1,1),model,20) = 0.15145419401392873

## 5. Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 6 decimal places):

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100,max_depth=50)

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=1693) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain] Xtest = X[idxtest] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() #xtrainscaled = scale.fit_transform(Xtrain) #xtestscaled = scale.fit_transform(Xtest) model.fit(Xtrain,ytrain) yhat = model.predict(Xtest) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

x = np.array(df['Apparent Temperature (C)'])

y = np.array(df['Humidity'])

DoKFold(x.reshape(-1,1),y.reshape(-1,1),model,10) = 0.14350325833172203

## 6. Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693)and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 5 decimal places):

from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=6)

model = LinearRegression()

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=1693) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain] Xtest = X[idxtest] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() x_poly_train = polynomial_features.fit_transform(np.array(Xtrain).reshape(-1,1)) x_poly_test = polynomial_features.fit_transform(np.array(Xtest).reshape(-1,1)) #xtrainscaled = scale.fit_transform(Xtrain) #xtestscaled = scale.fit_transform(Xtest) model.fit(x_poly_train,ytrain) yhat = model.predict(x_poly_test) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

x = np.array(df['Apparent Temperature (C)'])

y = np.array(df['Humidity'])

DoKFold(x.reshape(-1,1),y.reshape(-1,1),model,10) = 0.1434659719585773

## 7. If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):

model = Ridge(alpha = 0.2)

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=1234) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain] Xtest = X[idxtest] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() xtrainscaled = scale.fit_transform(Xtrain) xtestscaled = scale.fit_transform(Xtest) model.fit(xtrainscaled,ytrain) yhat = model.predict(xtestscaled) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

x = np.array(df['Temperature (C)'])

y = np.array(df['Humidity'])

DoKFold(x.reshape(-1,1),y.reshape(-1,1),model,10) = 0.1514458793622936

## 8. Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)' 'Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):

x1 = pd.DataFrame(df['Humidity'])

x2 = pd.DataFrame(df['Wind Speed (km/h)'])

x3 = pd.DataFrame(df['Pressure (millibars)'])

x4 = pd.DataFrame(df['Wind Bearing (degrees)'])

x = pd.concat([x1,x2,x3,x4], axis = 1)

y = np.array(df['Temperature (C)'])

x = np.array(x)

model = LinearRegression()

polynomial_features = PolynomialFeatures(degree=6)

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=1234) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain, :] Xtest = X[idxtest, :] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() x_poly_train = polynomial_features.fit_transform(np.array(Xtrain).reshape(-1,1)) x_poly_test = polynomial_features.fit_transform(np.array(Xtest).reshape(-1,1)) xtrainscaled = scale.fit_transform(x_poly_train) xtestscaled = scale.fit_transform(x_poly_test) model.fit(xtrainscaled,ytrain) yhat = model.predict(xtestscaled) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

DoKFold(x,y.reshape(-1,1), model, 10)

## 9. Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):

x1 = pd.DataFrame(df['Humidity'])

x2 = pd.DataFrame(df['Wind Speed (km/h)'])

x3 = pd.DataFrame(df['Pressure (millibars)'])

x4 = pd.DataFrame(df['Wind Bearing (degrees)'])

x = pd.concat([x1,x2,x3,x4], axis = 1)

y = np.array(df['Temperature (C)'])

x = np.array(x)

model = RandomForestRegressor(n_estimators = 100, max_depth= 50)

def DoKFold(X,y,model,k): PE2 = [] kf = KFold(n_splits=k,shuffle=True,random_state=1234) for idxtrain,idxtest in kf.split(X): Xtrain = X[idxtrain, :] Xtest = X[idxtest, :] ytrain = y[idxtrain] ytest = y[idxtest] scale = StandardScaler() #xtrainscaled = scale.fit_transform(Xtrain) #xtestscaled = scale.fit_transform(Xtest) model.fit(Xtrain,ytrain) yhat = model.predict(Xtest) PE2.append(np.sqrt(MSE(ytest,yhat))) return np.mean(PE2)

DoKFold(x,y.reshape(-1,1), model, 10) = 5.830546731164867

## 10. If we visualize a scatter plot for Temperature (on the horizontal axis) vs Humidity (on the vertical axis) the overall trend seems to be

import matplotlib.pyplot as plt

x = np.array(df['Temperature (C)'])

y = np.array(df['Humidity'])

plt.scatter(x,y)

<img width="399" alt="Screen Shot 2021-03-25 at 11 17 59 AM" src="https://user-images.githubusercontent.com/74326062/112497202-c4e7d480-8d5b-11eb-8a86-b702d90046fb.png">
