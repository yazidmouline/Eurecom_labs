import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, cos

PI = 3.141592653589793238462643383279502884197169

def sign(x):
    '''
    The sign function (returns -1 if x is negative and 1 if it is non-negative)
    '''
    if x < 0:
        return -1
    else:
        return 1

class Dataset:
    '''
    A class representing a dataset. The first layer of an instance of the MLP class should be a Dataset object.
    '''
    def __init__(self, datafile=None, input_size=0, length=0):
        self.input = []
        self.output = []
        if datafile:
            with open(datafile) as f: # Open the dataset file
                n_inputs = int(f.readline().strip()) # The first line gives the number of inputs per line
                self.input_size = n_inputs
                for line in f: # For each sample
                    sample = np.array([float(x) for x in line.strip().split()])
                    X = sample[:n_inputs]
                    self.input.append(X) # Append the inputs to self.input
                    self.output.append(sample[n_inputs]) # Append the outputs to self.output
            self.len = len(self.input) # Number of samples in the dataset (accessible through len(dataset))
        else:
            self.input_size = input_size
            self.len = length
        self.indices = list(range(self.len)) # List of indices used to pick samples in a random order

    def next_sample(self):
        '''
        Pick next sample
        '''
        i = random.choice(self.indices)
        return (self.input[i], self.output[i])

    def __len__(self):
        return self.len

class SVM:
    '''
    A class representing a Support Vector Machine
    '''
    def __init__(self, dataset, test_dataset, print_step=None, verbose=False): # infile: MLP description file, dataset: Dataset object
        self.verbose = verbose
        self.dataset = dataset # Current dataset
        self.train_dataset = dataset # Training dataset
        self.test_dataset = test_dataset # Testing dataset
        self.plot = list() # You can use this to make plots
        self.test_plot = list() # Plot for test dataset
        input_size = dataset.input_size
        self.w = np.random.rand(input_size+1)-0.5 # self.w[-1] is actually b
        self.print_step = print_step # Print accuracy during training every print_step
        sample, self.gt = dataset.next_sample() # Initialize input and output of MLP
        self.x = np.array(sample.tolist() + [1.])

    def __str__(self):
        res = "(w = "
        res += str(self.w[:-1])
        res += ", b = "
        res += str(self.w[-1])
        res += ")"
        return res

    def __call__(self):
        res = np.dot(self.w, self.x)
        return res

    def train_one_iteration(self, l, eta_t):
        '''
        Train for one epoch
        '''
        pass

    def train(self, n_iterations, l):
        '''
        Train function (with specified number of epochs, lambda and decay)
        '''
        if not self.print_step:
            self.print_step = max(1, int(n_iterations/50))
        # For n_iterations iterations...
        for i in range(n_iterations):
            # Set the learning rate to 1/((i+1)*l)...
            eta_t = 1/((i+1)*l)
            # And train for one iteration!
            self.train_one_iteration(l, eta_t)

            if not i%(self.print_step):
                if self.verbose:
                    print("Epoch:", i+1, "out of", n_iterations)
                    self.print_accuracy()
                else:
                    self.compute_accuracy()

    def setnextinput(self):
        '''
        Set input of SVM to next input of dataset
        '''
        sample, gt = self.dataset.next_sample()
        self.gt = gt
        self.x = np.array(sample.tolist() + [1.])

    def save_SVM(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def restore_SVM(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def setmode(self, mode):
        '''
        Function used to change between training set and testing set
        '''
        if mode == "train":
            self.dataset = self.train_dataset
        elif mode == "test":
            self.dataset = self.test_dataset
        else:
            print("Unknown mode!")

    def print_accuracy(self):
        '''
        Print accuracy of neural network on current dataset
        '''
        print("Accuracy:", 100*self.compute_accuracy(), "%")

    def compute_accuracy(self):
        '''
        Compute accuracy of neural network on train and test dataset and return accuracy on test dataset
        '''
        # Compute accuracy on training set
        self.setmode("train")
        n_samples = len(self.dataset)
        n_accurate = 0.
        self.dataset.index = 0
        for i in range(n_samples):
            self.setnextinput()
            if sign(self()) == self.gt: # self() returns dot(w, x)
                n_accurate += 1.
        self.plot.append(n_accurate/n_samples)

        # Compute accuracy on testing set
        self.setmode("test")
        n_samples = len(self.dataset)
        n_accurate = 0.
        self.dataset.index = 0
        for i in range(n_samples):
            self.setnextinput()
            if sign(self()) == self.gt:
                n_accurate += 1.
        self.test_plot.append(n_accurate/n_samples)

        # Do not forget to go back to the training set!
        self.setmode("train")

        return n_accurate/n_samples

    def reset_plot(self):
        '''
        Reset plot
        '''
        self.plot = list()
        self.test_plot = list()

    def make_plot(self):
        '''
        Print plot
        '''
        plt.plot([x*self.print_step for x in range(len(self.plot))], self.plot, 'r-')
        plt.plot([x*self.print_step for x in range(len(self.test_plot))], self.test_plot, 'g-')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.axis([0, self.print_step*(len(self.plot)-1), 0, 1.05])
        plt.show()


def print_dataset(data_file):
    '''
    A function printing a plot of a 2D dataset (no need to understand it)
    '''
    datasets = ["data/dataset" + str(x) + ".txt" for x in range(1, 5)]
    if data_file not in datasets:
        print("Non-printable dataset!")
        return
    else:
        print("Dataset:", data_file)

    points = []

    with open(data_file) as f:
        f.readline()
        for line in f:
            x, y, z = [float(a) for a in line.strip().split()]
            points.append(np.array([x, y, z]))

    plt.plot([x[0] for x in points if x[2] == -1], [x[1] for x in points if x[2] == -1], 'r.')
    plt.plot([x[0] for x in points if x[2] == 1], [x[1] for x in points if x[2] == 1], 'g.')
    if data_file == datasets[0]:
        plt.plot([x/10-2 for x in range(40)], [-x/10+2 for x in range(40)], 'b--')
    elif data_file == datasets[1]:
        plt.plot([x/10-1.5 for x in range(30)], [0.5-((x/10)-1.5)**3 for x in range(30)], 'b--')
    elif data_file == datasets[2]:
        plt.plot([x/(4*17.321)-0.866 for x in range(120)]+[np.sqrt(3)/2], [np.sqrt(1.5-2*(x/(4*17.321)-0.866)**2) for x in range(120)]+[0], 'b--')
        plt.plot([x/(4*17.321)-0.866 for x in range(120)]+[np.sqrt(3)/2], [-np.sqrt(1.5-2*(x/(4*17.321)-0.866)**2) for x in range(120)]+[0], 'b--')
    elif data_file == datasets[3]:
        plt.plot([x/8 for x in range(17)] +
             [2 for x in range(17)] +
             [(2-x/8) for x in range(17)] +
             [0 for x in range(17)],
             [0 for y in range(17)] +
             [y/8 for y in range(17)] +
             [2 for y in range(17)] +
             [(2-y/8) for y in range(17)], 'b--')
    plt.show()