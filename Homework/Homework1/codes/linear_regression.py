import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pprint

from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split

file = open("CALI.csv")
data=pd.read_csv(file,)
df_x = pd.DataFrame(data, columns = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'])
df_y = pd.DataFrame(data,columns=['HOUSING PRICE'])

from statsmodels.api import OLS
model_LR = OLS(df_y, df_x).fit()
print(model_LR.summary())

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)
results = model.predict(x_test)
print("Here is the example of testing value")
print(np.c_[y_test.values, results][0:5,:])

from sklearn.metrics import mean_squared_error, r2_score
model_score = model.score(x_train,y_train)
print('R2 sq: ', model_score)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test, results))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_test, results))

pred_val = model_LR.fittedvalues.copy()
residual = pred_val - df_y.values.flatten()

fig, ax = plt.subplots(figsize=(6,2.5))
ax.scatter(residual, pred_val)

sns.distplot(residual)

import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)

# Examining simple linear regression...

#X = df_x['CRIM'].values.reshape(-1, 1)
X = df_x.iloc[:,0].values.reshape(-1, 1)
#X = df_x['RM'].values.reshape(-1, 1)
#X = df_x.iloc[:,5].values.reshape(-1, 1)
Y = df_y.values.reshape(-1,1)

model3 = linear_model.LinearRegression()
model3.fit(X,Y)
Y_pred = model3.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()