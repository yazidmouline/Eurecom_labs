import random
from math import exp
import matplotlib.pyplot as plt

def sigmoid(x):
    '''
    Sigmoid function
    '''
    res = 1/(1 + exp(-x))
    return res

def d_sigmoid(x):
    '''
    Derivative of the sigmoid function
    '''
    s = sigmoid(x)
    res = s*(1-s)
    return res

def argmax(l):
    '''
    Argmax function
    '''
    maxval = max(l)
    for i in range(len(l)):
        if l[i] == maxval:
            return i

class Neuron:
    '''
    A class representing a neuron
    '''
    def __init__(self, inputs):
        n_inputs = len(inputs) # "inputs" is either a list of floats or a Layer object
        self.inputs = inputs
        self.weights = []
        # Random weights initialization
        for _ in range(n_inputs):
            self.weights.append(random.random())
        # Random bias initialization
        self.bias = random.random()
        self.u = 0 # Activation of the neuron
        self.out = .5 # Output of the neuron
        self.d_weights = [0. for _ in range(n_inputs)] # Derivatives of the loss wrt weights
        self.d_u = 0. # Derivative of the activation

class Dataset:
    '''
    A class representing a dataset. The first layer of an instance of the MLP class should be a Dataset object.
    '''
    def __init__(self, datafile):
        self.input = []
        self.output = []
        self.index = 0
        maxval = 0.
        with open(datafile) as f: # Open the dataset file
            n_inputs = int(f.readline().strip()) # The first line gives the number of inputs per line
            for line in f: # For each sample
                sample = [float(x) for x in line.strip().split()]
                X = sample[:n_inputs]
                maxval = max(maxval, max(-min(X), max(X)))
                self.input.append(X) # Append the inputs to self.input
                self.output.append(sample[n_inputs:]) # Append the outputs to self.output
        # The following few lines of code aim at avoiding having too high or too low values as inputs
        for x in self.input:
            for i in range(len(x)):
                x[i] /= maxval
        self.len = len(self.input) # Number of samples in the dataset (accessible through len(dataset))
        self.indices = list(range(self.len)) # List of indices used to pick samples in a random order

    def next_sample(self):
        '''
        Pick next sample
        '''
        if self.index == self.len: # If we arrived at the end of the dataset...
            self.index = 0 # then reset the pointer
            random.shuffle(self.indices) # and shuffle the dataset
        i = self.index
        i = self.indices[i]
        self.index += 1
        return (list(self.input[i]), list(self.output[i]))

    def __len__(self):
        return self.len

class Layer:
    '''
    A class representing a layer
    '''
    def __init__(self, inlayer, n_neurons):
        self.len = n_neurons # Number of neurons in the layer (accessible through len(layer))
        self.neurons = [] # List of neurons of the layer
        self.input = inlayer # Previous layer (or a dataset)
        n_inputs = len(inlayer) # Number of inputs
        for _ in range(n_neurons):
            self.neurons.append(Neuron(inlayer)) # Initialize neurons

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        item = self.neurons[key].out
        return item

    def feedforward(self):
        '''
        Feedforward function (just apply the feedforward function of each neuron)
        '''
        for n in self.neurons:
            n.feedforward()

class MLP:
    '''
    A class representing a Multi-Layer Perceptron
    '''
    def __init__(self, infile, dataset, print_step=100, verbose=True): # infile: MLP description file, dataset: Dataset object
        self.verbose = verbose
        self.inputs = dataset
        self.plot = list() # You can use this to make plots
        self.print_step = print_step # Print accuracy during training every print_step
        sample, self.gt = dataset.next_sample() # Initialize input and output of MLP
        self.layers = [sample] # First layer of MLP: inputs
        with open(infile) as f:
            for line in f:
                n_units = int(line.strip())
                self.layers.append(Layer(self.layers[-1], n_units)) # Create other layers

    def feedforward(self):
        '''
        Feedforward function (just apply the feedforward function of each layer)
        '''
        for i in range(1, len(self.layers)):
            self.layers[i].feedforward()

    def __str__(self):
        sizes = list()
        for l in self.layers:
            sizes.append(len(l))
        return str(sizes)

    def backpropagate(self, learning_rate):
        '''
        Backpropagation function (with given learning rate)
        '''
        self.compute_gradients()
        self.apply_gradients(learning_rate)

    def compute_gradients(self):
        '''
        To be implemented
        '''
        pass

    def apply_gradients(self, learning_rate):
        '''
        To be implemented
        '''
        pass

    def train_one_epoch(self, learning_rate):
        '''
        Train for one epoch
        '''
        for i in range(len(self.inputs)):
            self.setnextinput() # Use next sample of dataset as next input
            self.feedforward() # Feed forward...
            self.backpropagate(learning_rate) # and backpropagate

    def train(self, n_epochs, learning_rate, decay=1.):
        '''
        Train function (with specified number of epochs, learning rate and decay)
        '''
        # previous_weights = []
        # for l in self.layers[1:]:
        #     for n in l.neurons:
        #         previous_weights.append(n.weights)
        for i in range(n_epochs):
            self.train_one_epoch(learning_rate)
            if not (i+1)%(self.print_step):
                if self.verbose:
                    print("Epoch:", i+1, "out of", n_epochs)
                    self.print_accuracy()
                else:
                    self.compute_accuracy()
            learning_rate *= decay
        # new_weights = []
        # for l in self.layers[1:]:
        #     for n in l.neurons:
        #         new_weights.append(n.weights)

    def setnextinput(self):
        '''
        Set input of MLP to next input of dataset
        '''
        sample, gt = self.inputs.next_sample()
        self.gt = gt
        for i in range(len(self.layers[0])):
            self.layers[0][i] = sample[i]

    def save_MLP(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def restore_MLP(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def print_accuracy(self):
        '''
        Print accuracy of neural network on current dataset
        '''
        print("Accuracy:", 100*self.compute_accuracy(), "%")

    def compute_accuracy(self):
        '''
        Compute accuracy of neural network on current dataset
        '''
        n_samples = len(self.inputs)
        n_accurate = 0.
        self.inputs.index = 0
        for i in range(n_samples):
            self.setnextinput()
            self.feedforward()
            if argmax(self.layers[-1]) == argmax(self.gt):
                n_accurate += 1.
        self.plot.append(n_accurate/n_samples)
        return n_accurate/n_samples

    def reset_plot(self):
        '''
        Reset plot
        '''
        self.plot = list()

    def make_plot(self):
        '''
        Print plot
        '''
        plt.plot([x*self.print_step for x in range(len(self.plot))], self.plot, 'ro')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.show()
        

    def setdataset(self, dataset):
        '''
        Set new dataset
        '''
        self.inputs = dataset
