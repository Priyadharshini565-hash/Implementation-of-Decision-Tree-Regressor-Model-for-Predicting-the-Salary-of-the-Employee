# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.  

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRIYADHARSHINI R
RegisterNumber:  212224040253
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:

### Data Head:
<img width="429" height="220" alt="image" src="https://github.com/user-attachments/assets/35fbed1b-9234-4ee7-9e43-8dbe0375a6e5" />

### Data Info:
<img width="780" height="258" alt="image" src="https://github.com/user-attachments/assets/504d50a9-63ee-401e-a36d-9e29575b888a" />

### isnull() sum():
<img width="296" height="105" alt="image" src="https://github.com/user-attachments/assets/7d6a2edf-bd75-4a7b-b49e-d539abb40763" />

### Data Head for salary:
<img width="391" height="216" alt="image" src="https://github.com/user-attachments/assets/600fe77f-06e6-44f7-8482-fc2802f568ea" />

### Mean Squared Error:
<img width="250" height="39" alt="image" src="https://github.com/user-attachments/assets/86dacd22-2723-474d-84d1-c633dfd8c6f4" />

### r2 Value:
<img width="356" height="33" alt="image" src="https://github.com/user-attachments/assets/a0b29a7f-9ffb-4002-a58d-4aa780063530" />

### Data prediction :
<img width="344" height="34" alt="image" src="https://github.com/user-attachments/assets/896f3c53-95b6-4b51-8ffe-8f6c799faa2a" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
