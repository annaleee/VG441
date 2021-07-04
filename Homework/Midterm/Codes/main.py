import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def answer_1_a(data):
    plt.scatter(np.linspace(0,59,60),data.demands)
    plt.show()

def answer_1_b(data):
    from statsmodels.tsa.stattools import adfuller
    from numpy import log
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(data.demands, order=(1, 1, 1))
    model_fit = model.fit(disp=0)
    model_fit.plot_predict(dynamic=False)
    plt.show()

def answer_1_c(data,n,depth,rate,loss):
    from sklearn.model_selection import train_test_split
    from sklearn import ensemble
    x_column=np.linspace(0,59,60)
    params = {'n_estimators': n, 'max_depth': depth, 'min_samples_split': 2, 'learning_rate': rate, 'loss': loss}
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(x_column.reshape(-1,1), data.demands)
    y_predicted = model.predict(x_column.reshape(-1,1))
    plt.scatter(x_column.reshape(-1,1), data.demands, edgecolors=(0, 0, 0))
    plt.plot(x_column.reshape(-1,1), y_predicted)
    plt.show()


if __name__ == '__main__':
    data=pd.read_csv("demands.csv",index_col=0)
    answer_1_c(data,50,1,1,'ls')