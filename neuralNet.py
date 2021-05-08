import numpy as np
import matplotlib.pyplot as plt


# based on 10.1109/ICASSP.2019.8682311
class Network():
    # initialize
    def __init__(self, kernel_size, input_size, kernel=np.array([]), C=np.array([]), weights=np.array([]), rates=[0.1,0.1,0.1]):
        self.K_SIZE = kernel_size if not kernel.any() else len(kernel)
        self.I_SIZE = input_size
        # check if weights for the Kernel are given, if not initialize random weights
        if not kernel.any():
            self.KERNEL = self.initialize_weights(kernel_size)
        else:
            self.KERNEL = kernel

        # check if C scaling parameter is given, if not set to random value
        if not C.size > 0:
            self.C = np.random.random()
        else:
            self.C = C

        # check if weights for fully connected layer are given, if not initialize random weights
        if not weights.any():
            self.WEIGHTS = self.initialize_weights(input_size - kernel_size+1)
        else:
            self.WEIGHTS = weights
        
        self.RATE_K, self.RATE_C, self.RATE_W = rates # learning rates

    # set the parameters of the model
    def set_parameters(self, kernel, c, weights):
        self.KERNEL = kernel
        self.K_SIZE = len(kernel)
        self.C = c
        self.WEIGHTS = weights

    # update learning rates, could be used for simulated annealing
    def set_rates(self, rateK, rateC, rateW):
        self.RATE_K, self.RATE_C, self.RATE_W = rateK, rateC, rateW

    # initialize normalized weights
    def initialize_weights(self, size):
        k = np.random.random(size)
        return k
    
    # Apply Softmax nonlinearity to data and scale with c
    def softmax(self, data):
        num = np.exp(self.C*(data-np.max(data)))
        denom = np.sum(np.exp(self.C*(data-np.max(data))))
        return num/denom

    # convolution layer convolves data with self.KERNEL
    def convolve(self, data):
        return np.convolve(data, self.KERNEL, 'valid')

    # fully connected layer
    def fully_connected(self, data):
        return np.dot(data, self.WEIGHTS)

    # run through network with given input
    def evaluate(self, inpt):
        out = []
        for i in range(inpt.shape[0]):
            out.append(self.fully_connected(self.softmax(self.convolve(inpt[i,:]))))
        return np.array(out)

    # run through network in forward direction, calculate local gradients
    def forward(self, inpt):
        self.CHI = self.convolve(inpt) # convolve spectrum with kernel
        self.PI = self.softmax(self.CHI) # nonlinear activation
        self.GRAD_PI_CHI = self.C*(np.diag(self.PI)-np.outer(self.PI, self.PI.T)) # local "gradient" of softmax with respect to CHI
        self.GRAD_PI_C = np.multiply(self.PI, self.CHI) - self.PI*np.dot(self.PI, self.CHI) # local "gradient" of softmax with respect to c
        self.GRAD_CHI_K = np.array([[inpt[i+j] for i in range(self.I_SIZE-self.K_SIZE+1)] for j in range(self.K_SIZE)]) # local "gradient" of convolution with kernel
        self.F0 = self.fully_connected(self.PI) # weighted average for fully connected layer
        return self.F0

    # backpropagation through the neural network, update parameters in the end
    def backprop(self, F0, F0train):
        df0 = (F0 - F0train)
        dw = df0*self.PI
        dpi = df0*self.WEIGHTS
        dc = np.dot(dpi, self.GRAD_PI_C)
        dchi = np.dot(dpi, self.GRAD_PI_CHI)
        dk = np.dot( self.GRAD_CHI_K, dchi)

        return dc, dw, dk
        #self.C -= self.RATE_C*dc
        #self.WEIGHTS -= self.RATE_W*dw
        #self.KERNEL -= self.RATE_K*dk
        
    # handles the training -- old, better use training module
    def train(self, dataIN, dataOUT):
        loss = np.inf
        while (loss > 4.0):
            loss = 0.0
            dc, dw, dk = 0.0, 0.0, 0.0
            for j in range(len(dataOUT)):
                f0 = self.forward(dataIN[j,:])
                dct, dwt, dkt = self.backprop(f0, dataOUT[j])
                dc += dct
                dw += dwt
                dk += dkt
                loss += 0.5*(f0 - dataOUT[j])**2 / len(dataOUT)
            self.C -= self.RATE_C * dc
            self.WEIGHTS -= self.RATE_W * dw
            self.KERNEL -= self.RATE_K * dk
            print(loss)
        return loss#

    # simple plotting procedure for visual verification
    def plot(self, dataset, trainY, offset=0.05):
        result = self.evaluate(dataset)
        fig, ax = plt.subplots(1,1)
        first = True
        for i in range(dataset.shape[0]):
            dat = dataset[i,:]
            if first:
                ax.plot(dat+i*offset, 'k', label='spectra')
                ax.plot([trainY[i]]*2, [dat[int(trainY[i])]+offset*i, offset*i], 'tab:blue', label='Peak Position')
                ax.plot([result[i]]*2, [dat[int(result[i])]+offset*i, offset*i], 'tab:orange', label='Estimation')
            else:
                ax.plot(dat+i*offset, 'k')
                ax.plot([trainY[i]]*2, [dat[int(trainY[i])]+offset*i, offset*i], 'tab:blue')
                ax.plot([result[i]]*2, [dat[int(result[i])]+offset*i, offset*i], 'tab:orange')
            first = False
        plt.legend(loc=0)





if __name__ == "__main__":
    # lorentzian for training data generation
    def lorentzian(x,x0,g,a):
        return a / (1.0+((x-x0)/g)**2) / np.pi / g

    training_sets = 20
    NN = Network(kernel_size=50, input_size=200, kernel=np.loadtxt('./kernel.txt'), C=np.loadtxt('./C.txt'), weights=np.loadtxt('./weights.txt'))
    lorentzian_positions = 150*np.random.random(training_sets)+25.0
    lorentzian_widths = 20*np.random.random(training_sets)+2.0
    lorentzian_heights = 1.0*np.random.random(training_sets)+0.5
    trainData = np.array([[lorentzian(x,lorentzian_positions[t], lorentzian_widths[t], lorentzian_heights[t])+0.01*np.random.random() for x in range(200)] for t in range(training_sets)])
    out = NN.train(trainData, lorentzian_positions)
    print("OUT: ", out)
    if not np.isnan(out):
        print("Saving")
        np.savetxt('./weights.txt', NN.WEIGHTS)
        np.savetxt('./C.txt', np.atleast_1d(np.array(NN.C)))
        np.savetxt('./kernel.txt', NN.KERNEL)
    

    # plot training sets
    output = NN.evaluate(trainData)
    offset=0.05
    fig, ax = plt.subplots(1,1)
    for i in range(training_sets):
        ax.plot(trainData[i,:]+offset*i)
        ax.scatter(output[i], offset*i)
        ax.scatter(lorentzian_positions[i], trainData[i,int(lorentzian_positions[i])]+offset*i)

    # plot new sets
    lorentzian_positions = 150*np.random.random(training_sets)+25.0
    lorentzian_widths = 20*np.random.random(training_sets)+2.0
    lorentzian_heights = 1.0*np.random.random(training_sets)+0.5
    inputData = np.array([[lorentzian(x,lorentzian_positions[t], lorentzian_widths[t], lorentzian_heights[t])+0.01*np.random.random() for x in range(200)] for t in range(training_sets)])
    outputNEW = NN.evaluate(inputData)
    offset=0.05
    fig2, ax2 = plt.subplots(1,1)
    for i in range(training_sets):
        ax2.plot(inputData[i,:]+offset*i)
        ax2.scatter(outputNEW[i], offset*i)

    # plot Kernel
    fig3, ax3 = plt.subplots(1,1)
    plt.plot(NN.KERNEL)
    plt.plot(NN.WEIGHTS)
    plt.show()