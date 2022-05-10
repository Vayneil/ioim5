import matplotlib.pyplot as plt
import numpy as np
import math

wb = load_workbook('./Dane.xlsx')
t = np.linspace(0.0, 10.0, 100000)
rho = []

for ws in wb.worksheets:
    rho_values = [ws.cell(i, 1).value for i in range(2, 100002)]
    T.append(ws.cell(6, 1))
    e_dot.append(ws.cell(6, 2))
    rho.append(e_values)
    

def objective(a):
  sum = 0
  for i in range(len(T)):
      for j in range(len(e[0])):
          err_squared = (sigma_test[i][j] - sigma_p(a, e[i][j], T[i], e_dot[i])) ** 2
          err_relative = err_squared / sigma_test[i][j]
          sum = sum + err_relative
  return sum
