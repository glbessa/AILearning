from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)

database = np.array([
    [1,1,1,0,0,0],
    [1,0,1,0,0,0],
    [1,1,1,0,0,0],
    [0,0,1,1,1,1],
    [0,0,1,1,0,1],
    [0,0,1,1,0,1]
])

#print(database.shape)

rbm.train(database, max_epochs=5000)

print(rbm.weights)

user1 = np.array([
    [1,1,0,1,0,0]
])

hidden_user1 = rbm.run_visible(user1)

user2 = np.array([
    [0,0,0,1,1,0]
])

hidden_user2 = rbm.run_visible(user2)

recomendacao_user1 = rbm.run_hidden(hidden_user1)
recomendacao_user2 = rbm.run_hidden(hidden_user2)

print(recomendacao_user1)
print(recomendacao_user2)