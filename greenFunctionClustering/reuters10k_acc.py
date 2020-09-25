import sys
import numpy as np
from sklearn import metrics
from sklearn.cluster import k_means
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils.linear_assignment_ import linear_assignment

def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size

Y = np.loadtxt('D:/Visual Studio 2017/c_c++_resource/greenFunctionClustering/greenFunctionClustering/reuters10kLabels.txt', dtype = np.int0)
X = np.loadtxt('D:/Visual Studio 2017/c_c++_resource/greenFunctionClustering/greenFunctionClustering/grad.txt')
##labels = np.loadtxt('D:/Visual Studio 2017/c_c++_resource/greenFunctionClustering/greenFunctionClustering/reuters10kDataAssessment.txt', dtype = np.int0)
centroids, labels, *_ = k_means(X = X, n_clusters = 4, n_init = 100)
print(cluster_acc(labels, Y), metrics.normalized_mutual_info_score(labels, Y))
with open('reuters10k_snn_knn_acc_nmi.txt', 'a') as fp:
    fp.write('SNN = %d\tKNN = %d:\t%f\t%f\n'%(int(sys.argv[1]), int(sys.argv[2]), cluster_acc(labels, Y), metrics.normalized_mutual_info_score(labels, Y)))