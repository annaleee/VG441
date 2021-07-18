import numpy as np
import gurobipy as gp
from gurobipy import GRB
S=np.array([[1,1,0,0,0,0,0,1],
            [1,0,0,0,1,1,0,0],
            [0,0,1,0,0,1,0,0],
            [0,1,0,0,0,0,1,1],
            [0,0,0,1,1,0,0,0]])

m=gp.Model()
x = m.addVars(5, lb=[0,0,0,0,0], vtype=GRB.CONTINUOUS, name="decision var")
m.setObjective(x[0]+x[1]+x[2]+x[3]+x[4], GRB.MINIMIZE)
c1 = m.addConstrs(1<=x[0]*S[0][t]+x[1]*S[1][t]+x[2]*S[2][t]+x[3]*S[3][t]+x[4]*S[4][t] for t in range(8))
m.optimize()
m.printAttr('X')