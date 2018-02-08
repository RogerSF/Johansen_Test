import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
from sklearn import linear_model as LM
from Johnsen_Test import coint_johansen as Jtest

style.use('ggplot')

EWA = pd.read_csv('EWA.csv')
EWC = pd.read_csv('EWC.csv')
P_A = EWA['Close'][:-1]
P_C = EWC['Close']
R_P_A = P_A.pct_change()[1:]
R_P_C = P_C.pct_change()[1:]

r1 = pd.DataFrame(P_A)
r2 = pd.DataFrame(P_C)
r1.fillna(0, inplace=True)
r2.fillna(0, inplace=True)

ret = pd.concat([r1, r2], axis=1)
print(ret.shape)

result = Jtest(ret)
print(result.evec[0])

MRP = result.evec[0][0] * P_A + result.evec[0][1] * P_C
MRP.plot()
plt.show()

#P_A.plot()
#P_C.plot()
#plt.show()
