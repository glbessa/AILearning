from rbm import RBM
import numpy as np

if __name__ == "__main__":
    movies = [
        'Freddy x Jason',
        'O Ultimato Bourne',
        'Star Trek',
        'Exterminador do Futuro',
        'Norbit',
        'Star Wars'
    ]
    ds = np.array([
        [0,1,1,1,0,1],
        [1,1,0,1,1,1],
        [0,1,0,1,0,1],
        [0,1,1,1,0,1],
        [1,1,0,1,0,1],
        [1,1,0,1,1,1]
    ])
    leo = np.array([
        [0,1,0,1,0,0]
    ])

    rbm = RBM(num_visible=len(movies), num_hidden=3)

    rbm.train(ds, max_epochs=5000)

    hidden_leo = rbm.run_visible(leo)
    recommendation_leo = rbm.run_hidden(hidden_leo)

    print(recommendation_leo)

    for i in range(len(leo[0])):
        if leo[0, i] == 0 and recommendation_leo[0,i] == 1:
            print(movies[i])

