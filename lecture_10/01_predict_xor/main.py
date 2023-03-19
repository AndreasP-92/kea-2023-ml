import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0,0], [0,1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

mlpc = MLPClassifier(max_iter=5000)
mlpc.fit(X, y)

print(mlpc.predict(X))

h = mlpc.predict(X)

print(h)