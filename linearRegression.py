import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sns.set_theme()

df = pd.read_csv("C:/Users/hasan/Desktop/Programming/Jupyter Notebook Files/Udemy Courses/Python For Data Science and Machine Learning Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv")

# print(df.info())

# sns.pairplot(df)
# plt.show()

# sns.distplot(df["Price"])
# plt.show()

# print(df.corr())
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
Y = df["Price"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, Y_train)
print(lm.intercept_)
print(lm.coef_)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"])
print("")
print(cdf)
print("")
predictions = lm.predict(X_test)
# print(predictions)

# plt.scatter(Y_test, predictions)
# plt.show()

# sns.distplot((Y_test - predictions))
# plt.show()

# Evaluating
print("")
print("MEAN ABSOLUTE ERROR")
print(metrics.mean_absolute_error(Y_test, predictions))

print("")
print("MEAN SQUARED ERROR")
print(np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
