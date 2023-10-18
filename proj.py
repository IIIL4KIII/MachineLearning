import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import seaborn as sns
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
meanData = X.iloc[:, :5]
meanData = pd.concat([meanData, y], axis=1)

from mpl_toolkits import mplot3d

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")


ax.scatter3D(X['mean radius'], X['mean perimeter'], X['mean area'], c=y, cmap='bwr', edgecolors='black')


 
# show plot
plt.show()