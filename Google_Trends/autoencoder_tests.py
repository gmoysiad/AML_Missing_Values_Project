import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from DTW.DTW import DynamicTimeWarping


class LSTMAE(tf.keras.Model):

    def __init__(self, *shape, neurons=128):
        super(LSTMAE, self).__init__()
        self.neurons = neurons
        self.shape = shape

    def __build__(self, inputs):
        self.il = tf.keras.layers.InputLayer(input_shape=(None, self.shape[0], self.shape[1]))
        self.lstm1 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(self.neurons//2, return_sequences=False)
        self.rp = tf.keras.layers.RepeatVector(self.shape[0])
        self.lstm3 = tf.keras.layers.LSTM(self.neurons//2, return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(self.neurons, return_sequences=True)
        self.td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.shape[1]))

    def call(self, inputs, training=None, mask=None):
        x = self.il(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.rp(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        return self.td(x)


class DenseAE(tf.keras.Model):

    def __init__(self, *shape, neurons=128):
        super(DenseAE, self).__init__()
        self.il = tf.keras.layers.InputLayer(input_shape=(None, shape[0], shape[1]))
        self.d1 = tf.keras.layers.Dense(neurons)
        self.d2 = tf.keras.layers.Dense(neurons//2)
        self.d3 = tf.keras.layers.Dense(shape[1])
        self.d4 = tf.keras.layers.Dense(neurons//2)
        self.d5 = tf.keras.layers.Dense(neurons)
        self.td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))

    def call(self, inputs, training=None, mask=None):
        x = self.il(inputs)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.td(x)


class Rescalers(tf.keras.Model):

    def __init__(self, *shape, neurons=128):
        super(Rescalers, self).__init__()
        self.il = tf.keras.layers.InputLayer(input_shape=(shape[0], shape[1]))
        self.lstm1 = tf.keras.layers.LSTM(neurons, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(neurons, return_sequences=True)
        self.d = tf.keras.layers.Dense(shape[1])

    def call(self, inputs, training=None, mask=None):
        x = self.il(inputs)
        x = self.lstm1(x)
        return self.lstm2(x)


def dtw(*time_series, bag_size=60):
    DTW = DynamicTimeWarping()
    DTW.get(time_series[0], time_series[1], bucket_size=bag_size)
    return DTW.D


def smp_gen(points, data, dict_, freq):
    mod = points % freq
    indices = np.arange(mod, years * days + 1, step=freq)
    for ind, i in enumerate(indices):
        aux = data[max(i - freq, 0):i]
        aux = np.array([0. if item < scale else item for item in aux])

        min_ = np.min(aux)
        max_ = np.max(aux)
        range_ = np.divide((np.subtract(new_max, new_min, dtype=np.float32)),
                           np.subtract(max_, min_, dtype=np.float32), dtype=np.float32)
        result = range_ * aux + new_max - np.multiply(min_, np.divide((new_max - new_min), (max_ - min_))) - 1.

        dict_[freq][max(i - freq, 0):i] = result

    return freq, dict_[freq]


def transform(data):
    for k in data.keys():
        data[k] = data[k][1:]
        data[k] = np.stack(data[k])
        data[k] = data[k].reshape(data[k].shape[0], data[k].shape[1], 1)
        data[k] = data[k].astype(np.float32)
    return data


def frequency(data, f):
    t = []
    for i in range(0+f, data.shape[0], f):
        t.append(data[i-f:i])
    t = np.array(t)
    t = np.stack(t)
    return t.reshape(t.shape[0], t.shape[1], 1)


# test = samples[49].reshape((samples[49].shape[0], samples[49].shape[1], 1))


# %%
# TESTING AUTOENCODER AND LSTM RESCALE
if __name__ == '__main__':

    years = 4
    days = 365
    total = years * days
    scale = 1.25
    xx = np.linspace(0, np.pi, num=total)
    
    actual_data = np.exp(0.05 * xx, dtype=np.float32)
    actual_data = actual_data * (0.05 * np.sin(years * xx, dtype=np.float32) +
                                 2 * np.cos(years * 12 * xx, dtype=np.float32) + 1.) + \
        1 * np.sin(years * 52 * xx, dtype=np.float32)
       
    real_data = actual_data * scale  # For plotting purposes
    
    noise = 0.2 * np.random.uniform(-0.5, 0.5, total)
    actual_data = actual_data + noise
    actual_data = actual_data * scale
    noise_plot = noise + np.max(actual_data)
    
    ii = 5
    new_max, new_min = 1, 0
    freqs = [i * 7 for i in range(1, ii+1, 1)]
    epochs = [i * 10 for i in range(ii, 0, -1)]
    units = 128
    
    ex_dict = {i: np.zeros((total,), dtype=np.float32) for i in freqs}
    
    res = dict(map(lambda p: smp_gen(total, actual_data, ex_dict, p), freqs))
    samples = {}
    for key in res.keys():
        modulo = years * days % key
        samples[key] = np.array(np.split(res[key], np.arange(modulo, years * days, step=key)))
       
    # %%
    # For visualizing the problem
    plot_range = (1000, 3000)
    for i, values in enumerate(res.values()):
        plt.plot(xx[plot_range[0]:plot_range[1]], values[plot_range[0]:plot_range[1]], label=str(freqs[i]))
    plt.plot(xx[plot_range[0]:plot_range[1]], actual_data[plot_range[0]:plot_range[1]], 'c', label='actual data')
    plt.plot(xx[plot_range[0]:plot_range[1]], noise_plot[plot_range[0]:plot_range[1]] * scale, 'r', label='noise(rescaled)')
    plt.plot(xx[plot_range[0]:plot_range[1]], real_data[plot_range[0]:plot_range[1]], 'k', label='real data(without noise)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
       
    # %%
    # TRANSFORM AND PREPARE DATA
    samples = transform(samples)
    test_data = samples.pop(7)  # data that we will use to rescale and create the input shapes of the NNs
    temp = noise.reshape(noise.shape[0], 1)  # NO REAL USE
    target_data = temp.reshape(temp.shape[0], 1, 1)  # data that will be the target for the LSTM
       
    # %%
    # STACKED TRAIN/FIT
    dense, lstm, rescalers, actual_data_freq = [], [], [], []
       
    for i, (key, v) in enumerate(samples.items()):
        print('inside for loop')
        dae = DenseAE(v.shape[1], v.shape[2])
        dae.compile(loss='mae', optimizer='adam')
        
        lae = LSTMAE(v.shape[1], v.shape[2])
        lae.compile(loss='mae', optimizer='adam')
        
        rescaler = Rescalers(v.shape[1], v.shape[2])
        rescaler.compile(loss='mse', optimizer='sgd')
        
        temp = frequency(target_data, key)
        actual_data_freq.append(frequency(actual_data, key))
        
        _ = dae.fit(v, v, epochs=epochs[i], verbose=0, shuffle=False)
        _ = lae.fit(v, v, epochs=epochs[i], verbose=0, shuffle=False)
        _ = rescaler.fit(v, temp, epochs=epochs[i], verbose=0, shuffle=False)
        
        dense.append(dae)
        lstm.append(lae)
        rescalers.append(rescaler)
    
    # %%
    # CHAINED RECONSTRUCTION AND RESCALING
    pr1, pr2, pr3 = [], [], []
    for m1, m2, m3, d in zip(dense, lstm, rescalers, actual_data_freq):
        print(d.shape)
        rec1 = m1.predict(d)
        print('after rec1', rec1.shape)
        rec2 = m2.predict(d)
        print('after rec2', rec2.shape)
        pr1.append(m3.predict(rec1))
        print('after pr1')
        pr2.append(m3.predict(rec2))
        print('after pr2')
        pr3.append(m3.predict(d))
        print('after pr3')
    
    # %%
    # COMMENTED OUT DUE TO LONG TIME TO COMPUTE
    # DTW COST
    # dtw1, dtw2, dtw3 = [], [], []
    # for a, b, c, d in zip(pr1, pr2, pr3, actual_data_freq):
    #     dtw1.append(dtw(d.reshape(-1, 1), a.reshape(-1, 1)))
    #     dtw2.append(dtw(d.reshape(-1, 1), b.reshape(-1, 1)))
    #     dtw3.append(dtw(d.reshape(-1, 1), c.reshape(-1, 1)))
