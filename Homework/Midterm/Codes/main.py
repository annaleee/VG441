import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def answer_1_a(data):
    data.plot()
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
    x_train, x_test, y_train, y_test = train_test_split(x_column.reshape(-1,1), data.demands, test_size=0.2)
    params = {'n_estimators': n, 'max_depth': depth, 'min_samples_split': 2, 'learning_rate': rate, 'loss': loss}
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    model_score = model.score(x_train, y_train)
    print('R2 sq: ', model_score)
    y_predicted = model.predict(x_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")
    plt.show()


if __name__ == '__main__':
    data=pd.read_csv("demands.csv",index_col=0)
    answer_1_c(data,50,1,1,'ls')