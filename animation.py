import matplotlib.pyplot as plt
import numpy as np

X = np.random.uniform(0,10,10)
Y = np.random.uniform(0,10,10)

fig = plt.figure(figsize=(10,8))
plt.ion()
plt.grid()

for i in range(len(X)):
    plt.scatter(X[i], Y[i], c='red', s=100)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()
    plt.pause(0.1)

plt.ioff()
plt.show()