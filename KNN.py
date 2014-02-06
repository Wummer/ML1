from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
from operator import itemgetter
from collections import Counter

"""
def gaus(x,mean,stdev):
	x = 1 / ((2*math.pi*(stdev)**2)**0.5)
	print x

	# the histogram of the data
	#n, bins, patches = plt.hist(dist, 50, normed=1, facecolor='g', alpha=0.75)
	
	#plt.title('Histogram mean=%s stdev=%s' %(mean, stdev))
	#plt.grid(True)
	#plt.show()

#gaus(10,-1,1)
#plot_gaus(0,2,1000)
#plot_gausplot_gaus(-1,1,1000)


def gaus2D():
	gausmean=[1,2]
	gauscovar=[[0.3,0.2],[0.2,0.2]]
	size = 100
	x,y = np.random.multivariate_normal(gausmean, gauscovar, size).T
	plt.plot(x,y,'x')
	plt.title('I.2.2 Multivariate gaussian distribution')
	plt.show()
#gaus2D()
"""
#-----------------------------------------------------------------------
#I.4

train = open('IrisTrain2014.dt', 'r')
test = open('IrisTest2014.dt', 'r')

"""
This function reads ind in the files, strips by newline and splits by space char. 
It returns he dataset as numpy arrays.
"""
def read_data(filename):
	data_set = ([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		data_set.append(l)	
	return data_set


def Euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them. 
	It returns the distance.
	"""
	inner = 0
	for i in range(len(ex1)-1): #We don't want the last value - that's the class
		inner += (ex1[i] - ex2[i])**2 
	distance = math.sqrt(inner)
	return distance

def NearestNeighbor(tr,ex0,K):
	"""
  	This function calls the euclidean and stores the distances with datapoint in a list of lists. 
  	These lists are sorted according to distances and K-nearest datapoints are returned 
	"""
	distances = []
	#distances.append(ex0)
	for ex in tr:
		curr_dist = Euclidean(ex,ex0) 
		distances.append([curr_dist,ex])

	distances.sort(key=itemgetter(0))
	KNN = distances[:K] #taking only the k-best matches
	return KNN

"""
This function calls KNN functions. I gets array (incl. class label) of KNN from NearestNeighbor-function. 
Most frequent class is counted. 
1-0 loss is calculated for train and test using counters. 
For train the 
"""	
def eval(train,test,K):
	correcttrain=0
	correcttest=0
	#train set
	for ex in train:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] == ex[-1]:
			correcttrain +=1
	#test set		
	for ex in test:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] == ex[-1]:
			correcttest +=1
	return correcttrain/len(train), correcttest/len(test)

"""
This function splits the train set in 5 (almost) equal sized splits. It returns a list of the
5 slices containg lists of datapoints
"""
def sfold(train,s):
	slices = [train[i::s] for i in xrange(s)]
	return slices


"""
After having decorated with *, this function gets a slice for testing and uses the rest for training.
First we choose test-set - that's easy.
Then for every test-set for as many folds as there are: use the remaining as train sets exept if it's the test set itself. 
Then we sum up the result for every run and average over them and print the result.  
"""
def crossval(folds):
	print '*'*45
	print '3-fold cross validation'
	print '*'*45
	
	slices = sfold(train_prime,folds)
	
	for k in Kcrossval:
		print "Number of neighbors \t%d" %k
		temp = 0
		listoffame =[]
		for f in xrange(folds):
			crossvaltest = slices[f]
			crossvaltrain =[]
			
			for i in xrange(folds):
				if i != f: 
					for elem in slices[i]:
						crossvaltrain.append(elem)
			acctrain, acctest = eval(crossvaltrain,crossvaltest,k)
			temp += acctest
		av_result = temp/folds
		print "Averaged result \t%1.4f" %av_result
		print "-"*45


#Calling KNN
#I.4.1 and for KBest I.4.2
K = [1,3,5]
Kbest = [15,17,21]

train_prime = read_data(train)
test_prime = read_data(test)

for k in K: #here you can switch between K or Kbest
	acctrain, acctest = eval(train_prime, test_prime,k)
	print "-"*45
	print "Number of neighbors: \t%d" %k
	print "Accuracy train: \t%1.4f" %acctrain
	print "Accuracy test: \t%1.4f" %acctest
print "-"*45

#Calling cross-validation
#I.4.2
Kcrossval = [1,3,5,7,9,11,13,15,17,21,25]
crossval(5)