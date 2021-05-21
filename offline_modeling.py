# https://www.tutorialspoint.com/matplotlib/matplotlib_pylab_module.htm
# https://medium.com/meta-design-ideas/linear-regression-by-using-gradient-descent-algorithm-your-first-step-towards-machine-learning-a9b9c0ec41b1

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# sa = -10
# sb = -10

# ga = 0.5
# gb = 1.0

# def f(a,b,x):
# 	return a*x+b

# def df_da(a,b,x):
# 	return x

# def df_db(a,b,x):
# 	return 1


sa = 10
sb = 5

ga = 5
gb = 2

def f(a,b,x):
	return a * np.sin(2 * np.pi * (1/10) * (x + b))

def df_da(a,b,x):
	return np.sin(2 * np.pi * (1/10) * (x + b))

def df_db(a,b,x):
	return a * np.cos(2 * np.pi * (1/10) * (x + b)) * 2 * np.pi * (1/10)


def err(a,b,points):
	e = 0
	N = float(len(points))
	for p in points:
		x = p[0]
		y = p[1]
		e += (f(a,b,x) + y)**2
	return e/N

def derr_da(a,b,points):
	da = 0
	N = float(len(points))
	for p in points:
		x = p[0]
		y = p[1]
		da += 2 * (f(a,b,x) + y) * df_da(a,b,x)
	return da/N

def derr_db(a,b,points):
	db = 0
	N = float(len(points))
	for p in points:
		x = p[0]
		y = p[1]
		db += 2 * (f(a,b,x) + y) * df_db(a,b,x)
	return db/N


def step_gradient(a,b,points,learning_rate):
	a -= (learning_rate * derr_da(a,b,points)) 
	b -= (learning_rate * derr_db(a,b,points))
	return [a,b]

def gradient_descent(a,b,points,learning_rate,num_iterations):
	for i in range(num_iterations):
		plot(a,b,points,i)
		a,b = step_gradient(a,b,points,learning_rate)
	return [a,b]

def plot(a,b,points,i):
	e = err(a,b,points)
	t = 'iteration={0:d} amplitude={1:+7.3f} shift={2:+7.3f} cost={3:7.3f}'.format(i,a,b,e)
	ax2.text(1,9,t,bbox = {'facecolor':'white', 'edgecolor':'white'})	
	
	ta.append(a)
	tb.append(b)
	te.append(e)
	path, = ax1.plot3D(ta, tb, te, 'r.-')

	x = np.linspace(0,10,100)
	y = f(a,b,x)
	line.set_data(x,y)

	if (i % 10 == 0):
		plt.draw()
		plt.pause(0.001)
	 


x = np.linspace(0,10,100)
y = f(ga,gb,x)
y += 0.1 * (np.random.rand(len(y)) - 0.5)
points = np.stack((x,y), axis = -1)

x = np.linspace(0,10,100)
y = f(ga,gb,x)

a = np.linspace(-10, 10)
b = np.linspace(-10, 10)
A,B = np.meshgrid(a, b)
E = err(A,B,points)

ta = []
tb = []
te = []


plt.rcParams['toolbar'] = 'None' 
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(A, B, E, cmap='viridis', edgecolor='none')
path, = ax1.plot3D(ta, tb, te, 'r.')
ax1.set_xlabel("amplitude")
ax1.set_ylabel("shift")
ax1.set_zlabel("cost")
ax1.set_title('Cost graph')

ax2 = fig.add_subplot(1,2,2)
line, = ax2.plot(x,y)
ax2.scatter(points.T[0], points.T[1])
ax2.set_xlabel("time")
ax2.set_ylabel("amplitude")
ax2.set_xlim(0,10)
ax2.set_ylim(-10,10)
ax2.set_title('noise and anti wave')

plt.ion()
plt.show()

a,b = gradient_descent(sa,sb,points,0.01,10000)
print(a,b)

plt.ioff()
plt.show()

