#----------------------------------------------------------------------#
#                   Steepest Descent Algorithm                         #  
#----------------------------------------------------------------------#
#
#----------------------------------------------------------------------#
#                         Presented by:                                #
#                Tomer Grossman & Oriel Somech                         #
#             Computer Science Students - 2'nd Year, CLB               #  
#----------------------------------------------------------------------#
#
#----------------------------------------------------------------------#
#                 !!  IMPORTANT PRE-RUNNING:  !!                       #  
#                   Install packages on your IDE                       #
#                         BEFORE Debbuging!                            #
#----------------------------------------------------------------------#

#----------------------------------------------------------------------#

import numpy as np
import random as r
from sympy import *

# Objective function
def func(x, y):
  return (x-3)**2 +(y-10)**2

def gradient_x(X):
  x, y = symbols('x y')
  grad = lambdify(x, diff((x-3)**2 +(y-10)**2, x))
  return grad(X)

def gradient_y(Y):
  x, y = symbols('x y')
  grad = lambdify(y, diff((x-3)**2 +(y-10)**2, y))
  return grad(Y)

def range_size():
  x_best = r.randrange(1, 100)
  y_best = r.randrange(1, 100)
  for i in range(1, 100):
      x = r.randrange(1, 100)
      y = r.randrange(1, 100)
      if x_best > gradient_x(x):
          x_best = gradient_x(x)
      if y_best > gradient_y(y):
          y_best = gradient_y(y)
  return x_best, y_best

x = np.array(range_size())
y = np.array(range_size())
TAU= 0.8; imperfection = 0.8    

while gradient_x(x[0]) != 0: 
  step = 0.25
  gradient = np.array(gradient_x(x[0]), gradient_y(x[1]))
  p = -gradient / ((gradient ** 2).sum() ** 0.5)
  m = gradient.dot(p)
  t = - imperfection * m
  while func(*x) - func(*(x + step * p)) < step * t:  
    step *= TAU
  fx = -(gradient_x(x))
  x = x + (step * fx)
  print(x[0])

while gradient_y(y[0]) != 0:  
  step = 0.25
  gradient = np.array(gradient_x(x[0]), gradient_y(x[1]))
  p = -gradient / ((gradient ** 2).sum() ** 0.5)
  m = gradient.dot(p); t = - imperfection * m
  while func(*y) - func(*(y + step * p)) < step * t:
      step *= TAU

  fy = -(gradient_y(y))
  y = y + (step * fy)
  print(x[0], y[0])

print(f"\nMinimum point is: \n(x = {x[0]}, y = {y[0]})")
#----------------------------------------------------------------------#
#
#---------------------------------------------------------------------- #
#                          All Right Reserved®                          #  
#---------------------------------------------------------------------- #