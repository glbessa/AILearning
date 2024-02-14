from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from pylab import pcolor, colorbar
import matplotlib.pyplot as plt

data = pd.read_csv('../../../datasets/wines.csv')

X = data.iloc[:, 1:].values
print(X.shape)
y = data.iloc[:, 0].values
print(y.shape)

normalizer = MinMaxScaler(feature_range=(0,1))
X = normalizer.fit_transform(X)

som_side = int((5 * (X.shape[0] ** (1/2))) ** (1/2))
print(som_side)

som = MiniSom(x = som_side, y = som_side, input_len=X.shape[1], sigma=1, learning_rate=0.5, random_seed=2)

som.random_weights_init(X)

som.train_random(data=X, num_iteration=100)

#print(som._weights)
#print(som._activation_map)
#print(som.activation_response(X))

plt.imshow(som.distance_map())
#plt.show()

w = som.winner(X[1])
#print(w)

markers = ['o', 's', 'D']
color = ['r', 'g', 'b']

y = y - 1

for i, x in enumerate(X):
    w = som.winner(x)
    #print(w)
    plt.plot(w[0], w[1], markers[y[i]], markerfacecolor = 'None', markersize=10, markeredgecolor = color[y[i]], markeredgewidth = 2)

plt.show()