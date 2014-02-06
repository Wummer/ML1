from __future__ import division
import numpy as np 
import pylab as plt


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
	plt.plot(Gauss(np.linspace(-5,5,50),m,s),label=(m,s))
plt.legend(loc='upper right')
plt.show()



""" I.2.2 Sampling a multivariate Gaussian distribution

Here we utilize the numpy function multivariate_normal to generate 100 samples.
Alternatively, we could use 
"""
def gsample():
	cov = np.array([[0.3,0.2],[0.2,0.2]],dtype=float)
	mean = np.array([1,2]).T
	"""
	#factorisation with cholesky and numpy - wooo
	L = np.linalg.cholesky(cov)

	norm = np.random.normal(size=100*3).reshape(3,100)
	rand = mean + np.dot(L, norm)
	return rand
	"""
	#Alternatively, we can use the numpy function to get multivariate
	return np.random.multivariate_normal(mean,cov,100).T


x,y = gsample() #saving for later
#plotting
plt.plot(x,y,'x')
plt.axis('equal')
plt.show()



""" I.2.3 Means.
We compute the mean via the equation 2.121 pp. 113 Bishop(2010).
Then we utilize the standard deviation function as to quantify the deviation.
""" 
def MAXIMUMLIKELIHOODOMGWTFLOL(x,y):
	MLx = sum(x)*1/len(x)
	MLy = sum(y)*1/len(y)

	return MLx,MLy

def std(x):
	return np.sqrt(np.mean(abs(x - x.mean())**2))

MLx,MLy = MAXIMUMLIKELIHOODOMGWTFLOL(x,y)
stdx = std(x)
stdy = std(y)

"""Values:
Stdx, stxy = (0.55480464532, 0.409711872631)
MLx,MLy = (1.02586990158,1.98927066925)
"""

plt.plot(x,y,'x',label='Data')
plt.plot(MLx,MLy,'o',label='Mean')
plt.legend(loc="lower right")
plt.show()



""" I.2.4 Covariance: The geometry of multivariate Gaussian distributions.
Equation 2.122
"""

def MLcov(x,y,MLx,MLy):
	assert len(x) == len(y),len(MLx) == len(MLy)
	new_x = []
	new_y = []
	"""
	for xn in x:
		new_x += (xn - MLx)(xn - MLx)
	for yn in y:
		new_y += (yn- MLy)(yn - MLy)
	"""

	EMLx = 1/len(x)*sum((x - MLx)(x - MLx).T)
	EMLy = 1/len(y)*sum((y- MLy)(y - MLy).T)

	return EMLx,

MLcov(x,y,MLx,MLy)