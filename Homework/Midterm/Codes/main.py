import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
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

def answer_3_c():
    import numpy as np
    d = [220, 155, 105, 90, 170, 210,290]
    T, K, h = 7, 1000, 1.2
    M = 10e5
    WW = Model()

    q = WW.addVars(T, lb=np.zeros(7), vtype=GRB.CONTINUOUS, name="order_quantity")
    x = WW.addVars(T, lb=np.zeros(7), vtype=GRB.CONTINUOUS, name="inventory_level")
    y = WW.addVars(T, vtype=GRB.BINARY, name="if_order")

    WW.setObjective(quicksum(K * y[t] + h * x[t] for t in range(T)), GRB.MINIMIZE)

    c1 = WW.addConstrs(q[t] <= M * y[t] for t in range(T))
    c2 = WW.addConstrs(x[t] == x[t - 1] + q[t] - d[t] for t in range(1, T))
    c3 = WW.addConstr(x[0] == q[0] - d[0])
    WW.optimize()
    WW.printAttr('X')

if __name__ == '__main__':
    data=pd.read_csv("demands.csv",index_col=0)
    answer_3_c()