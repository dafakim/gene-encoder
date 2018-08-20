import os
import setGPU

import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
from collections import Iterable
#hyperparameters
lr = 0.001
steps = 100
batch_size = 50
logs_path = './logs/nobatch.summary'
print("tf : {}".format(tf.__version__))
print("numpy : {}".format(np.__version__))

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def intersect_mtlb(a, b):
    a = list(flatten(a))
    b = list(flatten(b))
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    ia = ia[np.isin(a1,c)]
    ib = ib[np.isin(b1,c)]
    return c, ia, ib

def get_batch(arr, batch_size):
    #get a random number between 0~max-batch_size as the starting index for splicing
    start = np.random.random_integers(0, len(arr)-batch_size)
    return start

def compare_matrix(old, new):
    #assume a and b are numpy arrays, compare how many 1s changed to 0s and vice versa
    zerotoone = 0
    onetozero = 0
    assert len(old) == len(new), "matrix a b length not equal {1} {2}".format(len(old). len(new))

    for idx in range(len(old)):
        if old[idx] == new[idx]:
            continue
        elif old[idx] == 0:
            zerotoone += 1
        else:
            onetozero += 1

    return zerotoone, onetozero

#load mat files
f1 = sio.loadmat('../data/mutation_matrices.mat')
data = f1['m_cdata']

#import sample-cancer_type data U
X = data[0,0]['X'].toarray()
U = data[0,0]['F'].toarray()
#process U as an input
sumcol = np.sum(U, axis=0)
U = np.divide(U, sumcol)
#extract gene names
gene_name = data[0,0]['geneID']
gene_name = gene_name[0]

#import pathway-gene data V0
f2 = sio.loadmat('../data/gene_name_info.mat')
gname = f2['gene_name_chr']
gname = gname[:,0]
common_gene,ai,bi = intersect_mtlb(gene_name, gname)
X = X[:,ai]

f3 = sio.loadmat('../data/bipartite_PPI_module.mat')
module_idx = f3['module_idx']
#if pathway is not ppi comment this part out
module_idx = module_idx.toarray()
module_idx = module_idx[:,bi]

f4 = sio.loadmat('../data/ppiMatrixTF.mat')
gene_name_ge = f4['gene_name_ge']
gene_name_ge = gene_name_ge[:,0]
c,ai,bi = intersect_mtlb(common_gene, gene_name_ge)

X = X[:,ai].astype('float64')
V0 = module_idx[:,ai]
V0 = np.divide(V0,np.sum(V0))
V0 = np.transpose(V0)

N,K1 = U.shape
M,K2 = V0.shape


#import cancer_type-pathways data S
#S is initialized to matrix of 1s, shape of K1, K2 from U and V shape
S = np.ones((K1,K2))
U[U == 0] = 1e-6
V0[V0 == 0] = 1e-6
V = np.transpose(V0)
#initialize weights of network to cancertype-pathway(S) and pathway-gene(V)
weights = {
    'W1': tf.Variable(tf.get_variable("S", initializer=S)),
    'W2': tf.Variable(tf.get_variable("V", initializer=V))
}
#create encoder with above weights
def network(x):
    layer1 = tf.sigmoid(tf.matmul(x, weights['W1']))
    layer2 = tf.sigmoid(tf.matmul(layer1, weights['W2']))
    return layer2

#use tf.convert_to_tensor to convert the input matrix into a tensor that can be used for computation
'''
print("type of X {}\ntype of U {}\ntype of V0 {}\ntype of S{}".format(type(X[0][0]),type(U[0][0]),type(V0[0][0]),type(S[0][0])))
exit()
'''
#construct model
phld = tf.placeholder(tf.float64, shape=(4790,19)) #temporary values, need to be 1,19
pred_X = network(phld)
true_X = tf.placeholder(tf.float64, shape=(4790,11089)) #temporary, need to be 1,11089

#define loss and optimizer
loss = tf.reduce_mean(tf.pow(true_X - pred_X, 2))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
#training loop to minimize loss
init = tf.global_variables_initializer()
tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for i in range(steps+1):
        _, ls,summary = sess.run([optimizer,loss,merged_summary], feed_dict={phld: U, true_X: X})
        summary_writer.add_summary(summary, i)
        print ("Step" + str(i) + ", Batch loss= " + "{!s:.10s}".format(ls))
#summary operation added to show that this works, need to show val loss foremost
    '''
    newS = weights['W1'].eval()
    newV = weights['W2'].eval()
    print(newS, S)
    flatS = newS.flatten().astype(int)
    flatV = newV.flatten().astype(int)
    Sbin= np.bincount(flatS)
    truthS = (Sbin[0] + Sbin[1]) == len(flatS)
    Vbin = np.bincount(flatV)
    truthV = (Vbin[0] + Vbin[1]) == len(flatV)
    print("newS bin truth")
    print(truthS)
    print("newV bin truth")
    print(truthV)
    '''
#check how the output has changed

