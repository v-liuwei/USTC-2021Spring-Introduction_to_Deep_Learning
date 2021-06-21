import numpy as np
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt


default_params = {
    'to_fit': np.sin,
    'x_range': (0, 4*np.pi),
    'data_size': 10000,
    'train_ratio': 0.8,
    'random_state': 7,
    'epochs': 10,
    'batch_size': 10,
    'criterion': nn.MSELoss()
}


# Comparision 1: depth and width
# fix total number of parameters about equal to 1000 and keep the width of each layer the same
# fix activation='tanh', lr=0.001
# following are five setups about depths and widths(depths only considers hidden layers)
# setup1: neurons=[1, 333, 1] (depths=1)       #(params)=1000
# setup2: neurons=[1, *[30]*2, 1] (depths=2)   #(params)=1021
# setup3: neurons=[1, *[21]*3, 1] (depths=3)   #(params)=988
# setup4: neurons=[1, *[15]*5, 1] (depths=5)   #(params)=1006
# setup5: neurons=[1, *[10]*10, 1] (depths=10) #(params)=1021
# repeat each setup 10 times, record each test loss
result1_raw = {
    1: [0.09346, 0.095363, 0.12187, 0.111205, 0.096947, 0.107818, 0.093153, 0.09429, 0.095786, 0.103154],
    2: [0.029461, 0.05171, 0.1099, 0.083763, 0.040612, 0.003585, 0.065114, 0.030679, 0.042282, 0.078191],
    3: [0.000704, 0.000287, 0.027187, 0.007998, 0.000721, 0.001373, 0.011837, 0.002505, 0.001198, 0.001492],
    5: [0.000982, 0.000224, 0.000635, 0.000259, 0.002295, 0.000729, 0.0007, 0.001617, 0.000853, 0.000275],
    10: [0.013834, 0.001034, 0.017508, 0.015356, 0.000298, 0.032453, 0.001733, 0.000175, 0.000309, 0.001068]
}
result1_mean = {k: np.mean(np.sort(v)[1:-1]) for k, v in result1_raw.items()}
print('## Comparision 1: depth and width ##')
print(result1_mean)

fig, ax = plt.subplots()
ax.set_title('Comparision of depths(widths)')
ax.set_xlabel('depth')
ax.set_ylabel('log10(test loss)')
ax.plot(result1_mean.keys(), np.log10(list(result1_mean.values())), marker='x')
plt.show()


# Comparision 2: activation function
# fix model structure as neurons=[1, 20, 20, 1], fix lr=0.001
# there are seven optional activation functions:
# sigmoid, tanh, relu, leakyrelu, prelu, elu, softplus
# repeat run 10 times with each activation function
result2_raw = {
    'sigmoid': [0.320537, 0.316262, 0.323674, 0.349383, 0.315778, 0.350341, 0.328564, 0.33681, 0.323076, 0.34205],
    'tanh': [0.095256, 0.085254, 0.077649, 0.136754, 0.068252, 0.065716, 0.116979, 0.07541, 0.054631, 0.047086],
    'relu': [0.09296, 0.032559, 0.013368, 0.109147, 0.04267, 0.02434, 0.027428, 0.061975, 0.065116, 0.022743],
    'leakyrelu': [0.030271, 0.069982, 0.066474, 0.048033, 0.024873, 0.028954, 0.069328, 0.063356, 0.021585, 0.063143],
    'prelu': [0.035761, 0.051486, 0.005814, 0.014586, 0.012871, 0.048049, 0.066837, 0.004329, 0.00277, 0.076942],
    'elu': [0.041305, 0.004289, 0.01048, 0.030719, 0.036398, 0.033117, 0.030902, 0.017483, 0.079652, 0.058757],
    'softplus': [0.11326, 0.093088, 0.088635, 0.081033, 0.080066, 0.08392, 0.094294, 0.079997, 0.058466, 0.070052]
}
result2_mean = {k: np.mean(np.sort(v)[1:-1]) for k, v in result2_raw.items()}
result2_std = {k: np.std(v) for k, v in result2_raw.items()}
print("## Comparision 2: activation function ##")
print(result2_mean)

fig, ax = plt.subplots()
ax.set_title('Comparision of activation functions')
ax.set_xlabel('activation function')
ax.set_ylabel('test loss')
ax.set_xticks(np.arange(len(result2_mean)))
ax.set_xticklabels(result2_mean.keys())
ax.scatter(np.arange(len(result2_mean)), list(result2_mean.values()), marker='x')
plt.show()


# Comparision 3: learning rate
# fix model structure as neurons=[1, 20, 20, 1], fix activation = 'tanh'
# following are five lr setups:
# {10^k: k=-1, -1.5, -2, -2.5, -3, -3.5, -4}
# repeat each setup 10 times, record each test loss
result3_raw = {
    10**-1: [0.370254, 0.641283, 0.366993, 0.396108, 0.501332, 0.404945, 0.86187, 0.377093, 0.769633, 0.407323],
    10**-1.5: [0.022817, 0.021794, 0.027867, 0.034896, 0.067046, 0.131822, 0.027459, 0.042532, 0.01076, 0.019996],
    10**-2: [0.004666, 0.001577, 0.003348, 0.000552, 0.001009, 0.000547, 0.001086, 0.061021, 0.01173, 0.005232],
    10**-2.5: [0.002043, 0.003938, 0.002026, 9.1e-05, 0.001185, 0.000199, 0.000501, 0.000886, 0.000758, 0.004475],
    10**-3: [0.094007, 0.023405, 0.155973, 0.066218, 0.079838, 0.006589, 0.073311, 0.002712, 0.062652, 0.136194],
    10**-3.5: [0.3407, 0.352521, 0.328751, 0.330109, 0.347772, 0.318212, 0.10451, 0.135686, 0.130908, 0.120176],
    10**-4: [0.365907, 0.369544, 0.345024, 0.37749, 0.36385, 0.36186, 0.376415, 0.385564, 0.352906, 0.360354],
}
result3_mean = {k: np.mean(np.sort(v)[1:-1]) for k, v in result3_raw.items()}
result3_std = {k: np.std(v) for k, v in result3_raw.items()}
print("## Comparision 3: learning rate")
print(result3_mean)
print(result3_std)

fig, ax = plt.subplots()
ax.set_title('Comparision of learning rates')
ax.set_xlabel('-log10(lr)')
ax.set_ylabel('log10(test loss)')
ax.plot(-np.log10(list(result3_mean.keys())), np.log10(list(result3_mean.values())), marker='x')
plt.show()