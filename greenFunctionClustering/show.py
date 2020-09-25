import sys
import numpy as np
import matplotlib.pyplot as plt

def show(dataSet, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    #mark = ['.b', '.y', '.r', '.c', 'xb', '*m', '1r', '+c', '.m', '.g', '.k']
    mark = ['xb', '1g', '+r', '.c', '^b', '.y', '.r', '.c']
    if clusterAssment.max() + 1 > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1
    for i in range(numSamples):
        markIndex = int(clusterAssment[i])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    plt.close()

X = np.loadtxt(str(sys.argv[1]))
labels = np.loadtxt(str(sys.argv[2]))
show(X, labels)
