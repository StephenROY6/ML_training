# Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model  # 以下只將sklearn中會用到的模組與函數分別匯入，可以加快速度
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# import seaborn as sns > 沒用到可以不用import，可以加快速度

# Load the Boston dataset
diabetes=datasets.load_diabetes()

# X - feature vectors
# # y - Target values
# X_train = pd.read_csv("E:/國防醫學院 醫學系/課業/lab/ML/Smith ML/0000000000002329_training_diabetes_x_y_train (1).csv")
X_data = pd.read_csv("E:/國防醫學院 醫學系/課業/lab/ML/Smith ML/0000000000002329_training_diabetes_x_y_train (1).csv")
x_test = pd.read_csv("E:/國防醫學院 醫學系/課業/lab/ML/Smith ML/0000000000002329_test_diabetes_x_test.csv")  #pd.read_csv無此參數: columns = X_data.columns
# 使用X_data的列名作为DataFrame的列名，即使列名是空的，也会被覆盖
x_test.columns = X_data.columns[:x_test.shape[1]]  #感覺用X_data.columns[:-1]會更好

y = X_data['Y']
X = X_data.drop(['Y'], axis=1)

# 或是用以下的方式
# X = X_data.drop(['Y'], axis=1, inplace=False)
# y = X_data['Y']

print(X.shape)
print(y.shape)
# X=diabetes.data
# y=diabetes.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)

# Create linear regression objest
lin_reg=linear_model.LinearRegression()

# Train the model using trai and test data
lin_reg.fit(X_train,y_train)

# Presict values for X_test data
predicted = lin_reg.predict(X_test)
predicted2 = lin_reg.predict(x_test)

# Regression coefficients
print('\n Coefficients are:\n',lin_reg.coef_) #挑出最佳化的係數

# Intecept> 鍵入下一個模型需要!?
print('\nIntercept : ',lin_reg.intercept_)

# variance score: 1 means perfect prediction

print('Variance score: ',lin_reg.score(X_test, y_test))

# Mean Squared Erroe

print("Mean squared error: %.2f\n"
      % mean_squared_error(y_test, predicted))

# Original data of y_test
expected = y_test

# Plot a graph for expected and predicted values
plt.title('Linear Regression ( DIABETS Dataset)')
plt.scatter(expected,predicted,c='b',marker='.',s=36)
plt.plot(np.linspace(0, 330, 100),np.linspace(0, 330, 100), '--r', linewidth=2)

plt.show()

#reference: 
# file: https://github.com/ravising-h/Diabetes-Datasets-LinearRegression/tree/master
# code: https://github.com/syamkakarla98/Linear-Regression/blob/master/LinearRegression_DIABETES_Dataset.py
