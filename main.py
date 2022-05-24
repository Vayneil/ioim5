# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
# Get started with interactive Python!
# Supports Python Modules: builtins, math,pandas, scipy 
# matplotlib.pyplot, numpy, operator, processing, pygal, random, 
# re, string, time, turtle, urllib.request
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp

# python implementation of particle swarm optimization (PSO)
# minimizing rastrigin and sphere function
 
import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float
 
wb = load_workbook('./Dane.xlsx')

rho = []

for ws in wb.worksheets:
    rho_values = [ws.cell(i, 1).value for i in range(2, 100002)]
    T.append(ws.cell(6, 1))
    e_dot.append(ws.cell(6, 2))
    rho.append(rho_values)

def sigma_p(T=898.0, t_max=1.0, e_dot=1.0, a):
    global t_cr
    step = 0.001
    # e_dot = 1.0
    # t_max = 1.0
    # T = 898.0
    b = 0.25e-9
    D = 30.0
    mu = 45000.0
    Q = 238000.0
    rho_0 = 1e4
    R = 8.314
    Z = e_dot * math.exp(Q / R / T)
    rho = rho_0
    t_0 = 0.0
    t = t_0
    rho_values = []
    t_cr = t_max

    # a1 = 2.1
    # a2 = 176.0
    # a3 = 19.5
    # a4 = 0.000148
    # a5 = 151.0
    # a6 = 0.973
    # a7 = 5.77
    # a8 = 1.0
    # a9 = 0.0
    # a10 = 0.262
    # a11 = 0.0
    # a12 = 0.000605
    # a13 = 0.167

    tau = 1e6 * mu * b ** 2 * 0.5
    l = a1 * 1e-3 / (Z ** a[12])
    A1 = 1 / b / l
    A2 = a[1] * e_dot ** (-a[8]) * math.exp(-a[2] * 1e3 / R / T)
    A3 = a[3] * 3e10 * tau / D * math.exp(-a[4] * 1e3 / R / T)
    rho_cr = -a[10] * 1e13 + a[11] * 1e13 * Z ** a[9]
    is_critical = False

    def drho_dt():
        global is_critical
        global t_cr
        if not is_critical:
            if rho > rho_cr:
                t_cr = t
                is_critical = True
                t2 = t - t_cr
                round(t2, 3)
                step_number = round(t2 / step)
                X = rho_values[step_number]
            else:
                X = 0
        else:
            t2 = t - t_cr
            round(t2, 3)
            step_number = round(t2 / step)
            X = rho_values[step_number]
        result = A1 * e_dot - A2 * e_dot * rho - A3 * rho ** a[7] * X
        return result


    while True:
        rho = rho + step * drho_dt()
        t += step
        rho_values.append(rho)
        if t > t_max:
            break


#-------fitness functions---------
 
# rastrigin function
def fitness_rastrigin(position):
  fitnessVal = 0.0
  for i in range(len(position)):
    xi = position[i]
    fitnessVal += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
  return fitnessVal
 
#sphere function
def fitness_sphere(position):
    fitnessVal = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitnessVal += (xi*xi);
    return fitnessVal;
    
#objective function
def objective(pos):
    sum = 0
    for i in range(len(T)):
        t = np.linspace(0.0, e_dot[i], 100000)
        for j in range(len(t)):
            err_squared = (sigma_test[i][j] - sigma_p(T[i], 1/e_dot[i], e_dot[i], pos)) ** 2
            err_relative = err_squared / sigma_test[i][j]
            sum = sum + err_relative
    return sum


#-------------------------
 
#particle class
class Particle:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
 
    # initialize position of the particle with 0.0 value
    self.position = [0.0 for i in range(dim)]
 
     # initialize velocity of the particle with 0.0 value
    self.velocity = [0.0 for i in range(dim)]
 
    # initialize best particle position of the particle with 0.0 value
    self.best_part_pos = [0.0 for i in range(dim)]
 
    # loop dim times to calculate random position and velocity
    # range of position and velocity is [minx, max]
    for i in range(dim):
      self.position[i] = ((maxx[i] - minx[i]) *
        self.rnd.random() + minx[i])
      self.velocity[i] = ((maxx[i] - minx[i]) *
        self.rnd.random() + minx[i])
 
    # compute fitness of particle
    self.fitness = fitness(self.position) # curr fitness
 
    # initialize best position and fitness of this particle
    self.best_part_pos = copy.copy(self.position)
    self.best_part_fitnessVal = self.fitness # best fitness
 
# particle swarm optimization function
def pso(fitness, max_iter, n, dim, minx, maxx):
  # hyper parameters
  w = 0.729    # inertia
  c1 = 1.49445 # cognitive (particle)
  c2 = 1.49445 # social (swarm)
 
  rnd = random.Random(0)
 
  # create n random particles
  swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]
 
  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = [0.0 for i in range(dim)]
  best_swarm_fitnessVal = sys.float_info.max # swarm best
 
  # computer best particle of swarm and it's fitness
  for i in range(n): # check each particle
    if swarm[i].fitness < best_swarm_fitnessVal:
      best_swarm_fitnessVal = swarm[i].fitness
      best_swarm_pos = copy.copy(swarm[i].position)
 
  # main loop of pso
  Iter = 0
  while Iter < max_iter:
     
    # after every 10 iterations
    # print iteration number and best fitness value so far
    if Iter % 10 == 0 and Iter > 1:
      print("Iter = " + str(Iter) + " best fitness = %.3f" % best_swarm_fitnessVal)
 
    for i in range(n): # process each particle
       
      # compute new velocity of curr particle
      for k in range(dim):
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
     
        swarm[i].velocity[k] = (
                                 (w * swarm[i].velocity[k]) +
                                 (c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k])) + 
                                 (c2 * r2 * (best_swarm_pos[k] -swarm[i].position[k]))
                               ) 
 
 
        # if velocity[k] is not in [minx, max]
        # then clip it
        if swarm[i].velocity[k] < minx[k]:
          swarm[i].velocity[k] = minx[k]
        elif swarm[i].velocity[k] > maxx[k]:
          swarm[i].velocity[k] = maxx[k]
 
 
      # compute new position using new velocity
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
   
      # compute fitness of new position
      swarm[i].fitness = fitness(swarm[i].position)
 
      # is new position a new best for the particle?
      if swarm[i].fitness < swarm[i].best_part_fitnessVal:
        swarm[i].best_part_fitnessVal = swarm[i].fitness
        swarm[i].best_part_pos = copy.copy(swarm[i].position)
 
      # is new position a new best overall?
      if swarm[i].fitness < best_swarm_fitnessVal:
        best_swarm_fitnessVal = swarm[i].fitness
        best_swarm_pos = copy.copy(swarm[i].position)
     
    # for-each particle
    Iter += 1
  #end_while
  return best_swarm_pos
# end pso
 
 
#----------------------------
# Driver code for rastrigin function
 
# print("\nBegin particle swarm optimization on rastrigin function\n")
# dim = 3
# fitness = fitness_rastrigin
 
 
# print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
# print("Function has known min = 0.0 at (", end="")
# for i in range(dim-1):
#   print("0, ", end="")
# print("0)")
 
# num_particles = 50
# max_iter = 100
 
# print("Setting num_particles = " + str(num_particles))
# print("Setting max_iter    = " + str(max_iter))
# print("\nStarting PSO algorithm\n")
 
 
 
# best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)
 
# print("\nPSO completed\n")
# print("\nBest solution found:")
# print(["%.6f"%best_position[k] for k in range(dim)])
# fitnessVal = fitness(best_position)
# print("fitness of best solution = %.6f" % fitnessVal)
 
# print("\nEnd particle swarm for rastrigin function\n")
 
 
# print()
# print()
 
 
# # Driver code for Sphere function
# print("\nBegin particle swarm optimization on sphere function\n")
# dim = 3
# fitness = fitness_sphere
 
 
# print("Goal is to minimize sphere function in " + str(dim) + " variables")
# print("Function has known min = 0.0 at (", end="")
# for i in range(dim-1):
#   print("0, ", end="")
# print("0)")
 
# num_particles = 50
# max_iter = 100
 
# print("Setting num_particles = " + str(num_particles))
# print("Setting max_iter    = " + str(max_iter))
# print("\nStarting PSO algorithm\n")
 
 
 
# best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)
 
# print("\nPSO completed\n")
# print("\nBest solution found:")
# print(["%.6f"%best_position[k] for k in range(dim)])
# fitnessVal = fitness(best_position)
# print("fitness of best solution = %.6f" % fitnessVal)
 
# print("\nEnd particle swarm for sphere function\n")

dim = 13
fitness = objective
cons_min = [0, 10, 1, 0, 10, 0.1, 1, 10, 0, 0, 0, 0, 0]
cons_max = [10, 25000, 100, 0.1, 500, 1, 10, 10, 10, 1, 1, 0.1, 1]
num_particles = 50
max_iter = 100
best_position = pso(fitness, max_iter, num_particles, dim, cons_min, cons_max)
fitness_val = fitness(best_position)
