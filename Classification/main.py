import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
# Make classification data and get it ready
# Make 1000 samples

n_samples = 1000

X, y = make_circles(n_samples, 
                    noise = 0.03, 
                    random_state= 42)

# print(X[:5])
# print(y[:5])

# Make DataFrame of circle data
circles = pd.DataFrame({"X1" : X[:,0],
                        "X2": X[:,1],
                        "label" : y})

print(circles.head(10))
plt.scatter(x = X[:, 0], 
            y = X[:, 1],
            c = y, 
            cmap = plt.cm.RdYlBu)
plt.show()