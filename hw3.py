import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditional normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.data = dataset
        self.class_value = class_value
        self.mean = calc_mean(dataset, class_value)
        self.std = calc_std(dataset, class_value)

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """

        size = np.size(self.data[:, -1])
        t = self.data[:, -1] == self.class_value
        return np.sum(t) / size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = normal_pdf(x[0], self.mean[0], self.std[0]) * normal_pdf(x[1], self.mean[1], self.std[1])
        return likelihood

    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    

class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.data = dataset
        self.class_value = class_value
        self.mean = calc_mean(dataset, class_value)
        self.cov_matrix = calc_cov_matrix(dataset, class_value)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        size = np.size(self.data[:, -1])
        t = self.data[:, -1] == self.class_value
        return np.sum(t) / size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov_matrix)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
    
def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pi = np.pi
    e = np.e
    normal = 1/np.sqrt(2 * pi * std ** 2) * e ** ((-(x - mean) ** 2) / (2 * std ** 2))
    return normal

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covariance matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pi = np.pi
    d = np.size(mean)
    det = np.linalg.det(cov)
    e = np.e
    cov_inv = np.linalg.inv(cov)
    x_minus_mean = x[:-1] - mean
    exp_pow = np.dot(x_minus_mean.T, cov_inv)
    exp_pow = np.dot(exp_pow, x_minus_mean)
    distribution = 1
    distribution *= (2 * pi) ** (-d/2)
    distribution *= det ** (-0.5)
    distribution *= e ** (-0.5 * exp_pow)
    return distribution

####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.data = dataset
        self.class_value = class_value
        self.n_class_value = np.sum(dataset[:, -1] == class_value)
        self.V_j = count_unique(dataset[:, :-1])

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        size = np.size(self.data[:, -1])
        t = self.data[:, -1] == self.class_value
        return np.sum(t) / size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        i = 0
        d = self.data[:, -1] == self.class_value
        for j in x[:-1]:
            t = self.data[d, i]
            n_i_j = np.sum(t == j)
            likelihood *= EPSILLON if n_i_j == 0 else ((n_i_j + 1) / (self.n_class_value + self.V_j[i]))
            i += 1
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        return 0 if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x) else 1
    

def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correct_predictions = 0
    for row in testset:
        if map_classifier.predict(row) == row[-1]:
            correct_predictions += 1
    return correct_predictions / np.size(testset, 0)


def calc_mean (dataset, class_value):
    relation = dataset[:,-1] == class_value
    data = dataset[relation, :-1]
    mean = np.mean(data, 0)
    return mean


def calc_std(dataset, class_value):
    relation = dataset[:, -1] == class_value
    data = dataset[relation, :-1]
    std = np.std(data, 0)
    return std

def calc_cov_matrix(dataset, class_value):
    relation = dataset[:, -1] == class_value
    data = dataset[relation, :-1]
    cov_matrix = np.cov(data.T)
    return cov_matrix

def count_unique(dataset):
    unique = np.array([])
    for row in dataset.T:
        unique = np.append(unique, np.size(np.unique(row)))

    return unique


            
            
            
            
            
            
    