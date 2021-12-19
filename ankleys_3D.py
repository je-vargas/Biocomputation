from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt

def objective(x, y):
 return -20.0 * exp(-0.2 * sqrt(0.05 * (x**2 + y**2)))-exp(0.05 * (cos(2 * 
  pi * x)+cos(2 * pi * y)))


r_min, r_max = -32, 32
xaxis = arange(r_min, r_max, 2.0)
yaxis = arange(r_min, r_max, 2.0)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)
figure = plt.figure()
axis = figure.gca( projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
plt.title("Akley's 3D")
plt.show()
# plt.contour(x,y,results)
# plt.show()
# plt.scatter(x, y, results)
# plt.show()