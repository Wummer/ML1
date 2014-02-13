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
#plt.legend(loc='upper right')
plt.show()



""" I.2.2 Sampling a multivariate Gaussian distribution

Here we utilize the numpy function multivariate_normal to generate 100 samples.
Alternatively, we could use 
"""
def gsample():
	cov = np.array([[0.3,0.2],[0.2,0.2]],dtype=float)
	mean = np.array([1,2]).T

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
def MaxLike(x,y):
	MLx = sum(x)*1/len(x)
	MLy = sum(y)*1/len(y)

	return MLx,MLy
# QUANTIFICATION

ML = MaxLike(x,y)

x_dev = (1-ML[0])/1
y_dev = (2-ML[1])/2

plt.plot(x,y,'x',label='Data')
plt.plot(ML[0],ML[1],'o',label='Sample Mean')
plt.plot(1,2,'ro', label="Distribution Mean")
plt.legend(loc="lower right")
plt.show()


""" I.2.4 Covariance: The geometry of multivariate Gaussian distributions.
Equation 2.122
"""

def MLcov(x,y,ML):
	assert len(x) == len(y),len(MLx) == len(MLy)

	samples = []
	nM  = 0
	MML = np.asarray([ML])

	for i in range(len(x)):
		samples.append([x[i],y[i]])
	samples = np.asmatrix(samples)

	for i in range(len(x)):
		n = samples[i]-MML
		nM += np.dot(n.T,n)

	EML = (1/len(x))*nM

	return EML

def transeig(ML,eigw,eigv):
	allteigs = []
	for i in range(len(eigw)):
		sqeig = np.dot(eigw[i],eigv[:,i])
		sqeig = np.sqrt(sqeig)
		teig = ML + sqeig
		allteigs.append(teig)

	return allteigs


def rotation(EML):
	"do something"


EML = MLcov(x,y,ML)

eigw, eigv = np.linalg.eig(EML)

teig = transeig(ML,eigw,eigv)
