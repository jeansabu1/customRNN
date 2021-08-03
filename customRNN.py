import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

import tensorflow as tf
from tensorflow.keras.layers import Dense

# Create a dataset
series = np.sin((0.1*np.arange(100))) + np.random.randn(100)*0.05


class seqData:
    def __init__(self, series, D=1, K=1, n_T=10):
        self.n_inputs = D
        self.n_outputs = K
        self.n_hidden = 128
        self.n_F = 1
        self.n_sample = n_T
        self.lenSeq = len(series)
        
        Xtmp = []
        Ytmp = []
        
        for t in range(self.lenSeq - self.n_sample):
            x = series[t:t+self.n_sample]
            Xtmp.append(x)
            y = series[t+self.n_sample]
            Ytmp.append(y)

        self.X = np.array(Xtmp).reshape(-1, self.n_sample)
        self.Y = np.array(Ytmp)
        self.N = len(self.X)

        self.X = self.X.astype(np.float32)
        self.T = self.Y.astype(np.float32)


class CustomRNN(tf.keras.Model):
    def __init__(self, seqData):
        super(CustomRNN, self).__init__()
        self.ht  = tf.Variable(tf.random_normal_initializer()(shape=[seqData.n_inputs, seqData.n_hidden]))
        self.Wxf = tf.Variable(tf.random_normal_initializer()(shape=[seqData.n_sample, seqData.n_hidden]))
        self.Whf = tf.Variable(tf.random_normal_initializer()(shape=[seqData.n_hidden, seqData.n_hidden]))
        self.Who = tf.Variable(tf.random_normal_initializer()(shape=[seqData.n_hidden, seqData.n_sample]))
        self.bh = tf.Variable(tf.zeros(seqData.n_hidden))
        self.bo = tf.Variable(tf.zeros(seqData.n_sample))
        self.fc = tf.Variable(tf.random_normal_initializer()(shape=[seqData.n_sample, seqData.n_outputs]))
        self.params = [self.Wxf, self.Whf, self.Who, self.bh, self.bo, self.fc]
        
    def call(self, inputs):
        self.ht = tf.sigmoid(tf.matmul(inputs, self.Wxf) + tf.matmul(self.ht, self.Whf) + self.bh)
        ot = tf.tanh(tf.matmul(self.ht, self.Who)+self.bo)
        Ypred = tf.matmul(ot, self.fc)
        return Ypred


class Learning:
    def __init__(self, _model, _seqData, lr=0.0001):
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        self.model = _model
        self.seqData = _seqData
        self.losses = []
        #self.numEpochs = args.numEpochs

    def get_loss(self, inputs, targets):
        predictions = self.model(inputs)
        error = targets -predictions
        return tf.reduce_mean(tf.square(error))
    
    def get_grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.get_loss(inputs, targets)
            return tape.gradient(loss_value, self.model.params)
            
    def train(self):
        for i in range(args.numEpochs):
            loss = 0
            for j in range(self.seqData.lenSeq - self.seqData.n_sample):
                X_flat = self.seqData.X[j,:].reshape(-1, self.seqData.n_sample)
                grads = self.get_grad(X_flat, self.seqData.Y[j])
                self.optimizer.apply_gradients(zip(grads, self.model.params))
                loss += self.get_loss(X_flat, self.seqData.Y[j])
        
            self.losses.append(loss)
            print(" iteration  ", i, " loss :: ", loss)

_D = 1
_K = 1
_n_T = 10

# Create the custom RNN model
seqData = seqData(series, _D, _K, _n_T)
model = CustomRNN(seqData)
learning = Learning(model, seqData, 0.0001)

def main(args):
    learning.train()
    plt.plot(learning.losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('custom RNN')
    parser.add_argument('--numEpochs', type=int, default=50)
    args = parser.parse_args()
    main(args)    


