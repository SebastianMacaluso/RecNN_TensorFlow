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
##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''
data_dir='data/'
os.system('mkdir -p '+data_dir)

pad=False

if pad:
  batches_dir='input_batches_pad/'
else:
  batches_dir='input_batches_no_pad/'

os.system('mkdir -p '+data_dir+'/'+batches_dir)
batch_filename='tt_QCD_jet'

myN_jets=5
batch_size=1


dir_jets_subjets='data/inputTrees' #+batch_label

sg='tt'
bg='qcd'
# -------------------------


sig_tree, sig_list=tree_list_batch.makeTrees(dir_jets_subjets,sg,myN_jets,0)
bkg_tree, bkg_list=tree_list_batch.makeTrees(dir_jets_subjets,bg,myN_jets,0)

# sig_label=np.ones((len(sig_list)),dtype=int)
# bkg_label=np.zeros((len(bkg_list)),dtype=int)
# 
# jet_list=np.concatenate((sig_list,bkg_list))
# label_list=np.concatenate((sig_label,bkg_label))


# print('jet_list label =',[jet['label'] for jet in jet_list])
# print('label_list=',label_list)
# indices = check_random_state(123).permutation(len(jet_list))
# print('indices=',indices)
# # jet_list = [jet_list[i] for i in indices]
# jet_list = jet_list[indices]
# label_list = label_list[indices]
# print('jet_list label =',[jet['label'] for jet in jet_list])
# print('label_list=',label_list)

train_x, train_y, dev_x, dev_y, test_x, test_y = batching.split_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)





# sys.exit()
# levels, children, n_inners, contents= batching.batch(sig_list)
# print('------'*20) 
# print('------'*20) 
# levels, children, n_inners, contents= batching.batch_Seb(train_x)
# sys.exit()

N_batches=len(train_x)//batch_size

# print('N_batches=',N_batches)

# tot_levels=[]
# tot_children=[]
# tot_n_inners=[]
# tot_contents=[]
# train_batches=[]


if pad==True:
  train_batches=batching.batch_array(train_x, train_y, batch_size)
  dev_batches=batching.batch_array(dev_x, dev_y, batch_size)
  test_batches=batching.batch_array(test_x, test_y, batch_size)

else:
  train_batches=batching.batch_array_no_pad(train_x, train_y, batch_size)
  dev_batches=batching.batch_array_no_pad(dev_x, dev_y, batch_size)
  test_batches=batching.batch_array_no_pad(test_x, test_y, batch_size)



# for i in range(batch_size,len(train_x)+1,batch_size): #This way even the last batch has the same size (We lose a few events at the end, with N_events<batch_size)
#   train_batches.append([])
#   levels, children, n_inners, contents= batching.batch_Seb(train_x[i-batch_size:i])
#   train_batches[-1].append(levels)
#   train_batches[-1].append(children)
#   train_batches[-1].append(n_inners)
#   train_batches[-1].append(contents)
# 
# 
# train_batches=np.asarray(train_batches)


# print('train_batches=',train_batches)

print('train_batches=',len(train_batches))
print('train_batches=',len(dev_batches))
# print('train_batches=',test_batches)

# Output the dataset to an npz file
np.savez_compressed(data_dir+batches_dir+batch_filename, train_batches=train_batches, dev_batches=dev_batches, test_batches=test_batches )