from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
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
	gausmean=[1,2].T
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
It returns he dataset as numpy arrays with.
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
	distance = np.sqrt(inner)
	return distance

def NearestNeighbor(tr,ex0,K):
	"""
  	This function expects a dataset, a datapoint and number of neighbors. 
  	It calls the euclidean and stores the distances with datapoint in a list of lists. 
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
1-0 loss and accuracy is calculated for train and test using counters. 
For the train accuracy I train on train and use datapoints from the same set.
For the test acc I train on train and use datapoints from test. 
"""	
def eval(train,test,K):
	wrongtrain=0
	wrongtest=0
	#train set
	for ex in train:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		bestresult = result.most_common(1)
		if bestresult[0][0] != ex[-1]:
			wrongtrain +=1

	#test set		
	for ex in test:
		ex_prime=NearestNeighbor(train,ex,K)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] != ex[-1]:
			wrongtest +=1
	return wrongtrain/len(train), wrongtest/len(test)

"""
This function splits the train set in 5 equal sized splits. It returns a list of the
5 slices containg lists of datapoints.
"""
def sfold(data,s):
	#np.random.shuffle(data) #shuffling doesn't change the result very much
	slices = [data[i::s] for i in xrange(s)]
	return slices


"""
After having decorated, this function gets a slice for testing and uses the rest for training.
First we choose test-set - that's easy.
Then for every test-set for as many folds as there are: use the remaining as train sets exept if it's the test set. 
Then we sum up the result for every run and average over them and print the result.  
"""
def crossval(trainset, folds):
	print '*'*45
	print '%d-fold cross validation' %folds
	print '*'*45

	slices = sfold(trainset,folds)

	for k in Kcrossval:
		print "Number of neighbors \t%d" %k
		temp = 0
		for f in xrange(folds):
			countsame = 0
			crossvaltest = slices[f]
			crossvaltrain =[]
			
			for i in xrange(folds):
				if i != f: 
					#print "this split is test: %d and this split is train %d" %(f,i) #only for debugging. It seems ok. It does not test on train slices
					for elem in slices[i]:
						crossvaltrain.append(elem)
			"""			
			#the following is for debuggin	
			for elem in crossvaltest:
				for e in crossvaltrain:
					if str(elem) == str(e):
						print "We are the same %s and %s" %(str(e), str(elem)) #The printed datapoint really are repeated datapoints in the train set
						countsame +=1
			"""
			
			acctrain, acctest = eval(crossvaltrain,crossvaltest,k)
			temp += acctest
			#print "Cross eval: number of same %d" %countsame	
		av_result = temp/folds
		print "Averaged 0-1 loss \t%1.4f" %av_result
		print "-"*45

#Computing mean and variance
"""
This function takes a dataset and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Mean = []
	Variance = []
	for e in data:
		number_of_features = len(e) - 1 #Leaving out the class

	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s +=elem[i]
		mean = s / len(data)
		Mean.append(mean)

		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(variance)
	return Mean, Variance


"""
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set is returned
"""
def meanfree(data):
	for e in data:
		number_of_features = len(e) - 1 #Leaving out the class


#Calling read and split
train_set = read_data(train)
test_set = read_data(test)

zeromean_train = meanfree(train_set)
zeromean_test = meanfree(test_set)

#Different K
K = [1,3,5]
Kcrossval = [1,3,5,7,9,11,13,15,17,21,25]
Kbest = [15,17,21]
Kbest2 = [1,11,15,21]

#Calling KNN
for k in Kcrossval: #here you can switch between different lists of K: K, Kcrosscal, Kbest, Kbest2
	acctrain, acctest = eval(zeromean_train, zeromean_test,k) # switch between datasets: train_set, test_set, zeromean_train, zeromean_test  
	print "-"*45
	print "Number of neighbors: \t%d" %k
	print "0-1 loss train:\t%1.4f" %acctrain
	print "0-1-loss test:\t%1.4f" %acctest
print "-"*45

# Calling crossval
crossval(zeromean_train, 5) #Switch between zeromean_train and train_set

