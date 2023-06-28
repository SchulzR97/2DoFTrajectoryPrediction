import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import datetime
import pickle

class RecurrentNetwork(object):
    """
    Class implementing a recurrent network with readout weights and RLS learning rules.

    **Parameters:**

    * `Ni` : Number of input neurons
    * `N` : Number of recurrent neurons
    * `No` : Number of read-out neurons
    * `tau` : Time constant of the neurons
    * `g` : Synaptic strength scaling
    * `pc` : Connection probability
    * `Io` : Noise variance
    * `P_plastic` : Percentage of neurons receiving plastic synapses
    """
    def __init__(self, Ni=2, N=800, No=1, tau=10.0, g=1.5, pc=0.1, Io=0.001, delta=1.0, P_plastic=0.6):
        
        # Copy the parameters
        self.Ni = Ni
        self.N = N
        self.No = No
        self.tau = tau
        self.g = g
        self.pc = pc
        self.Io = Io
        self.P_plastic = P_plastic
        self.N_plastic = int(self.P_plastic*self.N) # Number of plastic cells = 480

        # Input
        self.I = np.zeros((self.Ni, 1))

        # Recurrent population
        self.x = np.random.uniform(-1.0, 1.0, (self.N, 1))
        self.r = np.tanh(self.x)

        # Read-out population
        self.z = np.zeros((self.No, 1))

        # Weights between the input and recurrent units
        self.W_in = np.random.randn(self.N, self.Ni)

        # Weights between the recurrent units
        self.W_rec = (np.random.randn(self.N, self.N) * self.g/np.sqrt(self.pc*self.N))

        # The connection pattern is sparse with p=0.1
        connectivity_mask = np.random.binomial(1, self.pc, (self.N, self.N))
        connectivity_mask[np.diag_indices(self.N)] = 0
        self.W_rec *= connectivity_mask

        # Store the pre-synaptic neurons to each plastic neuron
        self.W_plastic = [list(np.nonzero(connectivity_mask[i, :])[0]) for i in range(self.N_plastic)]

        # Inverse correlation matrix of inputs for learning recurrent weights
        self.P = [np.identity(len(self.W_plastic[i])) for i in range(self.N_plastic)]

        # Output weights
        self.W_out = (np.random.randn(self.No, self.N) / np.sqrt(self.N))

        # Inverse correlation matrix of inputs for learning readout weights
        self.P_out = [np.identity(self.N) for i in range(self.No)]
    
    def LeakyReLU(self, x):
        out = x
        out[x < 0] = x[x < 0] * 0.01
        return out
    
    def ReLU(self, x):
        out = x
        out[x < 0] = 0.
        return out
        
    def reinitialize_readout_weights(self):
        "Reinitializes the readout weights while preserving the recurrent weights."

        # Output weights
        self.W_out = (np.random.randn(self.No, self.N) / np.sqrt(self.N))

        # Inverse correlation matrix of inputs for learning readout weights
        self.P_out = [np.identity(self.N) for i in range(self.No)]
        
    def reset(self):
        """
        Resets the activity in the network.
        """    
        self.x = np.random.uniform(-1.0, 1.0, (self.N, 1))
        self.r = np.tanh(self.x)
        self.z = np.zeros((self.No, 1))

    def step(self, I, noise=True):
        """
        Updates neural variables for a single simulation step.
        
        * `I`: input at time t, numpy array of shape (Ni, 1)
        * `noise`: if noise should be added to the recurrent neurons dynamics (should be False when recording the initial trajectory).
        """
        
        # Noise can be shut off
        I_noise = (self.Io * np.random.randn(self.N, 1) if noise else 0.0)
        
        # tau * dx/dt + x = I + W_rec * r + I_noise
        self.x += (np.dot(self.W_in, I) + np.dot(self.W_rec, self.r) + I_noise - self.x)/self.tau
        
        # r = tanh(x)
        self.r = np.tanh(self.x)
        #self.r = self.ReLU(self.x)
        
        # z = W_out * r
        self.z = np.dot(self.W_out, self.r)

    def train_recurrent(self, target):
        """
        Applies the RLS learning rule to the recurrent weights.
        
        * `target`: desired trajectory at time t, numpy array of shape (N, 1)
        """
        # Compute the error of the recurrent neurons
        error = self.r - target

        # Apply the FORCE learning rule to the recurrent weights
        for i in range(self.N_plastic): # for each plastic post neuron
            
            # Get the rates from the plastic synapses only
            r_plastic = self.r[self.W_plastic[i]]
            
            # Multiply the inverse correlation matrix P*R with the rates from the plastic synapses only
            PxR = np.dot(self.P[i], self.r[self.W_plastic[i]])
            
            # Normalization term 1 + R'*P*R
            RxPxR = (1. + np.dot(r_plastic.T,  PxR))
            
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P[i] -= np.dot(PxR, PxR.T)/RxPxR
            
            # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
            self.W_rec[i, self.W_plastic[i]] -= error[i, 0] * (PxR/RxPxR)[:, 0]

    def train_readout(self, target):
        """
        Applies the RLS learning rule to the readout weights.
        
        * `target`: desired output at time t, numpy array of shape (No, 1)
        """
        # Compute the error of the output neurons
        error = self.z - target

        # Apply the FORCE learning rule to the readout weights
        for i in range(self.No): # for each readout neuron
            
            # Multiply the rates with the inverse correlation matrix P*R
            PxR = np.dot(self.P_out[i], self.r)
            
            # Normalization term 1 + R'*P*R
            RxPxR = (1. + np.dot(self.r.T,  PxR))
            
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P_out[i] -= np.dot(PxR, PxR.T)/RxPxR
            
            # Learning rule W <- W - e * (P*R)/(1+R'*P*R)
            self.W_out[i, :] -= error[i, 0] * (PxR/RxPxR)[:, 0]

        return error
    
    def save(self, filename:str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            print('Net saved as {0}'.format(filename))
        return

        start = datetime.datetime.now()
        W_plastic_list_0 = []
        for l in self.W_plastic:
            W_plastic_list_1 = []
            for e in l:
                W_plastic_list_1.append(float(e))
            W_plastic_list_0.append(W_plastic_list_1)
                
        dict = {
            'Ni': float(self.Ni),
            'N': float(self.N),
            'No': float(self.No),
            'tau': self.tau,
            'g': self.g,
            'Io': self.Io,
            'pc': self.pc,
            'Io': self.Io,
            'N_plastic': float(self.N_plastic),
            'P_plastic': self.P_plastic,
            'W_in': np.ndarray.tolist(self.W_in),
            'W_plastic': W_plastic_list_0,
            'W_rec': np.ndarray.tolist(self.W_rec),
            'W_out': np.ndarray.tolist(self.W_out),
            'P': [p.tolist() for p in self.P],
            'P_out': [p_out.tolist() for p_out in self.P_out]
        }

        f = open(filename, mode='w')
        json.dump(dict, f)
        f.close()
        end = datetime.datetime.now()
        
def load(filename):
    with open(filename, 'rb') as f:
        net : RecurrentNetwork = pickle.load(f)
        print('Net {0}  loaded.'.format(filename))
    return net

    f = open(filename, mode='r')
    dict = json.load(f)
    net = RecurrentNetwork(
        Ni = int(dict['Ni']),
        N = int(dict['N']),
        No = int(dict['No']),
        tau = dict['tau'],
        g = dict['g'],
        pc = dict['pc'],
        Io = dict['Io'],
        P_plastic = dict['P_plastic'])
    net.W_in = np.array(dict['W_in'])
    net.W_plastic = []
    net.W_rec = np.array(dict['W_rec'])
    net.W_out = np.array(dict['W_out'])

    for l0 in dict['W_plastic']:
        W_plastic_list_1 = []
        for l1 in l0:
            W_plastic_list_1.append(int(l1))
        net.W_plastic.append(W_plastic_list_1)

    net.P = [np.identity(len(net.W_plastic[i])) for i in range(net.N_plastic)]
    net.P_out = [np.identity(net.N) for i in range(net.No)]

    f.close()
    return net

class Neuron():
    def __init__(self, tau, noise = 0.):
        self.x = 0.
        self.r = 0.
        self.noise = noise
        self.tau = tau

    def step(self, I):
        dx = (I + self.r + self.noise - self.x) / self.tau
        self.x += dx
        self.r = np.tanh(self.x)
        return self.x, self.r
    
if __name__ == '__main__':
    neuron = Neuron(tau=10)

    I = np.zeros(1000)
    I[10:20] = 1.
    I[400:500] = 1.

    X = []
    R = []

    for t in range(I.shape[0]):
        x, r = neuron.step(I[t])

        X.append(x)
        R.append(r)
    
    plt.subplot(2, 1, 1)
    plt.title('Impulse')
    plt.plot(I)

    plt.subplot(2, 1, 2)
    plt.title('Neuron Activity')
    plt.plot(X, label='x')
    plt.plot(R, label='r')
    plt.legend()
    plt.show()