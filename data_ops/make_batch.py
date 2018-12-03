# This file loads tre_list_batch.py and batching_tf.py and creates the minibatches for training


import pickle
import gzip
import sys
import os
import subprocess
#import matplotlib as mpl
import numpy as np
import json
import itertools
import re
import random
# import random
# random.seed(0)

from sklearn.utils import check_random_state
from data_ops import batching_tf as batching
import tree_list_batch 




sig_tree, sig_list=tree_list_batch.makeTrees(dir_jets_subjets,sg,myN_jets,0)
bkg_tree, bkg_list=tree_list_batch.makeTrees(dir_jets_subjets,bg,myN_jets,0)

sig_label=[1 for i in len(sig_list)]
bg_label=[0 for i in len(bkg_list)]

jet_list=np.concatenate((sig_list,bg_list))
label_list=np.concatenate((sig_label,bkg_label))

indices = check_random_state(123).permutation(len(jet_list))
print('indices=',indices)
X = [X[i] for i in indices]
y = y[indices]

print('X=',X)
print('y=',y)

sys.exit()
levels, children, n_inners, contents= batching.batch(sig_list) 

batch_size=128
N_batches=myN_jets//batch_size

tot_levels=[]
tot_children=[]
tot_n_inners=[]
tot_contents=[]
#   
#   for i in range(N_batches,myN_jets,batch_size):
#     levels, children, n_inners, contents= batching.batch(sig_list[i:i+batch_size])
#     tot_levels.append(levels)
#     tot_children.append(children)
#     tot_n_inners.append(n_inners)
#     tot_contents.append(contents)
  
#   print('levels=',levels)
#   print('---'*20)
#   print('children=',children)
#   print('---'*20)
#   print('n_inners=',n_inners)
#   print('---'*20)
#   print('contents=',contents)