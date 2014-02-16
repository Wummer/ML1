from __future__ import division
from math import *
import numpy as np 
import pylab as plt

"""

------------------------------------ I.2.x ---------------------------------------

"""


""" I.2.1	Univariate Gaussian Distribution
Here we calculate the Gaussian distributions from section 2.3.
We try with (mean,sigma) as: (-1,1),(0,2), and (2,3) on a linearly spaced distribution.
 """
def Gauss(x,m=-1,s=1):
	norm = 1 / ((2.*3.14*s**2.)**0.5)
	interesting = np.exp(- (1/(2*s**2.)) *(x-m)**2.)
	result = norm * interesting

	return result	

#plotting
for m, s in [(-1,1),(0,2),(2,3)]:
	plt.plot(Gauss(np.linspace(-5,5,50),m,s),label=('$%d,%d$')%(m,s))
plt.legend(loc='upper right')
plt.show()



""" I.2.2 Sampling a multivariate Gaussian distribution

Here we utilize the numpy function multivariate_normal to generate 100 samples.
"""
def gsample():
	cov = np.array([[0.3,0.2],[0.2,0.2]],dtype=float)
	mean = np.array([1,2])

	return np.random.multivariate_normal(mean,cov,100).T


x,y = gsample() #saving for later
plt.plot(x,y,'x')
plt.axis('equal')
plt.show()



""" I.2.3 Means.
We compute the mean via the equation 2.121 pp. 113 Bishop(2010).
Then we utilize the standard deviation function as to quantify the deviation.
""" 
def MaxLike(x,y):
	MLx = sum(x)*1/len(x)
	MLy = sum(y)*1/len(y)

	return MLx,MLy

ML = np.array(MaxLike(x,y)).reshape(2,1)

x_dev = (1-ML[0])/1
y_dev = (2-ML[1])/2

plt.plot(x,y,'x',label='$Data$')
plt.plot(ML[0],ML[1],'o',label='$Sample Mean$')
plt.plot(1,2,'ro', label="$Distribution Mean$")
plt.legend(loc="lower right")
plt.title('Deviation for x: %1.4f and y: %1.4f'%(x_dev,y_dev))
plt.axis('equal')
plt.show()


""" I.2.4 Covariance: The geometry of multivariate Gaussian distributions.
Equation 2.122
"""


"""
	Here we create the sample covariance matrix as it is described by Bishop in equation 2.122.
	To do this we utilize the previously computed sample mean.
"""
def MLcov(x,y,ML):
	assert len(x) == len(y)
	samples = []
	nM  = 0

	for i in range(len(x)):
		samples.append(np.array([x[i],y[i]]).reshape(2,1)) #2 columns, 1 row, i.e. vector plots
	samples = np.array(samples)

	for i in range(len(x)):
		n = samples[i]-ML
		nM += np.dot(n,n.T)
	CML = (1/len(x))*nM

	return CML


"""
	Here we scale and translate the eigenvector with the equation given in assignment pdf @ I.4.2.
"""
def transeig(ML,eigw,eigv):
	allteigs = []

	for i in range(len(eigw)):
		sqeig = np.sqrt(eigw[i])*eigv[:,i].reshape(2,1) #Python notation is weird - we're just adding two vectors together here as in [a,b]+[c,d] = [a+b,c+d]
		teig = ML + sqeig
		allteigs.append(teig)

	return allteigs

"""
	Here we create the rotation matrix and generate the new rotated covariance matrix rEML
"""

def rotation(CML,theta):
	R = np.array([[cos(theta),-sin(theta)],
				[sin(theta),cos(theta)]])

	rEML = np.linalg.inv(R)*CML*R

	return rEML

"""
	Here we solve the issue of finding the correct theta so that distribution spreads along the x-axis.
	The intuition being that when the larger eigenvector's y coordinate is equal to the means y coordinate then the distrubtion is spreading along the x-axis.
"""

def findtheta(ML,CML):
	for t in range(360):
		th = np.radians(i)
		rEML = rotation(CML,th)
		eigw,eigv = np.linalg.eig(rEML)
		teig = transeig(ML,eigw,eigv)





""" Calling the CML function and acquiring the initial eigenvectors & transformed eigenvectors 
CML = MLcov(x,y,ML) """

CML = MLcov(x,y,ML)
eigw, eigv = np.linalg.eig(CML)
teig = transeig(ML,eigw,eigv)
t1 = teig[0].tolist()
t2 = teig[1].tolist()


"""Plotting data and scaled & translated eigenvectors """
plt.plot(x,y,'x')
plt.arrow(float(ML[0]),float(ML[1]),float(t1[0]-ML[0]),float(t1[1]-ML[1]),fc="k", ec="k",head_width=0.05, head_length=0.1)
plt.arrow(float(ML[0]),float(ML[1]),float(t2[0]-ML[0]),float(t2[1]-ML[1]),fc="k", ec="k",head_width=0.05, head_length=0.1)
plt.axis('equal')
plt.show()


""" Rotating the gaussian sample. We resample the gaussian to acquire a rotated distribution.
This gives us entirely new data points that has been rotated and share almost the same mean as the original datapoints  """
degrees = [30,60,90]
new_x = x
new_y = y
plt.plot(x,y,'x')

for elem in degrees:
	r = radians(elem)
	rEML = rotation(CML,r)
	new_x,new_y = np.random.multivariate_normal([1,2],rEML,100).T
	plt.plot(new_x,new_y,'o')

plt.legend(['$\\theta=0$','$\\theta=30$',"$\\theta=60$","$\\theta=90$"],loc='best')
plt.axis()
plt.show()


""" 
	Finding the correct theta and plotting the datapoints + the vectors"""
"""

------------------------------------ I.4.x ---------------------------------------

"""