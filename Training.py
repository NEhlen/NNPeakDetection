import numpy as np
import matplotlib.pyplot as plt
import neuralNet as nn


# lorentzian for training data generation
def lorentzian(x,x0,g,a):
    return a / (1.0+((x-x0)/g)**2) / np.pi / g

# create a spectrum made from lorentzian peak plus noise
def create_spectrum(num_pixels):
    pos = 0.75*num_pixels*np.random.random()+0.1*num_pixels
    width = 0.05*num_pixels*np.random.random()+0.01*num_pixels
    height = 1.0*np.random.random()+1.0
    return np.array([lorentzian(x,pos, width, height)+0.025*np.random.random() for x in range(num_pixels)]), pos

# handles the training of Neural Network
class Training():
    def __init__(self, neuralnetwork):
        self.NN = neuralnetwork
        self.NPIX = self.NN.I_SIZE

    # generate a set of N training spectra
    def generate_training_set(self, N):
        spectrums, positions = [], []
        for i in range(N):
            s, p = create_spectrum(self.NPIX)
            spectrums.append(s)
            positions.append(p)
        return np.array(spectrums), np.array(positions)

    # train the neural network, train spectra in traindata 2D-np.array(), actual peak positions in trainresults 1D-np.array(), settings in settings dict
    def train(self, traindata, trainresults, settings={'mode' : 'batch', 'batch_size' : 20, 'iterations' : 100, 'save' : True}):
        # batch training
        if settings['mode'] == 'batch':
            bsize = settings['batch_size']
            iterations = settings['iterations']
            # split data in batches and run through all
            for i in range(np.int(np.floor(np.int(traindata.shape[0])/bsize))):
                dat = traindata[i*bsize:i*bsize+bsize, :] # data batch
                datres = trainresults[i*bsize:i*bsize+bsize]

                # optimize loop for batch
                loss = np.inf
                for k in range(iterations):
                    loss = 0.0
                    dc, dw, dk = 0.0, 0.0, 0.0
                    for j in range(bsize):
                        f0 = self.NN.forward(dat[j,:])
                        dct, dwt, dkt = self.NN.backprop(f0, datres[j])
                        dc += dct
                        dw += dwt
                        dk += dkt
                        loss += 0.5*(f0 - datres[j])**2 / len(datres)
                    self.NN.C -= self.NN.RATE_C * dc
                    self.NN.WEIGHTS -= self.NN.RATE_W * dw
                    self.NN.KERNEL -= self.NN.RATE_K * dk
                    print(loss)
                    if loss < 0.5:
                        break
        # online mode
        elif settings['mode'] == 'online':
            # repeated iterations
            for i in range(settings['iterations']):
                loss = np.inf

                # loop through training data
                for k in range(traindata.shape[0]):
                    loss = 0.0
                    
                    f0 = self.NN.forward(traindata[k,:])
                    dc, dw, dk = self.NN.backprop(f0, trainresults[k])
                    loss += 0.5*(f0 - trainresults[k])**2
                    self.NN.C -= self.NN.RATE_C * dc
                    self.NN.WEIGHTS -= self.NN.RATE_W * dw
                    self.NN.KERNEL -= self.NN.RATE_K * dk
                    print(loss)
                    if loss < 0.01:
                        break
        if not np.isnan(loss) and settings['save']:
            print("Saving")
            np.savetxt('./weights.txt', self.NN.WEIGHTS)
            np.savetxt('./C.txt', np.atleast_1d(np.array(self.NN.C)))
            np.savetxt('./kernel.txt', self.NN.KERNEL)
        return loss
            

if __name__ == "__main__":
    NN = nn.Network(kernel_size=50, input_size=200, kernel=np.loadtxt('./kernel.txt'), C=np.loadtxt('./C.txt'), weights=np.loadtxt('./weights.txt'), rates=[0.05, 0.05, 0.05])
    Train = Training(NN)
    #Train.train(*Train.generate_training_set(1000), settings={'mode' : 'batch', 'batch_size' : 50, 'iterations' : 100, 'save': True})
    NN.plot(*Train.generate_training_set(20), offset=0.05) # plot test data
    plt.figure()
    plt.plot(NN.KERNEL)
    plt.plot(NN.WEIGHTS)
plt.show()