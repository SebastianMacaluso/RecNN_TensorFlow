##---------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------
#LOAD LIBRARIES
from __future__ import print_function
import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pickle
import gzip
import subprocess
#import matplotlib as mpl
import json
import itertools
import re

import utils
import tree_batch as tree 

np.set_printoptions(threshold=np.nan)

import scipy 

import time
start_time = time.time()

import image_preprocess_functions_color as fns

random.seed(0)

import os
import numpy as np
import tensorflow as tf

import tree_batch as tree 

##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

# myN_jets=1000000000
myN_jets=20

sg='tt'
bg='qcd'

dir_jets_subjets='inputTrees' #+batch_label






MODEL_STR = 'rnn_embed=%d_l2=%f_lr=%f.weights'
SAVE_DIR = './weights/'

###############################################################################
# CLASSES
###############################################################################


class Config(object):
  """Holds model hyperparams and data information.
  Model objects are passed a Config() object at instantiation.
  """
  mygpu='/gpu:0'
  embed_size = 35
  label_size = 2
  early_stopping = 2
  anneal_threshold = 0.99
  anneal_by = 1.5
  max_epochs = 1
  lr = 0.01
  l2 = 0.02
  
  batch_size=10
  capacity = 20*batch_size
  min_after_dequeue=10 * batch_size
  
  model_name = MODEL_STR % (embed_size, l2, lr)
  
###############################################################################
# FUNCTIONS AND CLASSES
###############################################################################
#8) Split the sample into train, cross-validation and test
def split_sample(sig, bkg, train_frac_rel, val_frac_rel, test_frac_rel):
  
  print('---'*20)
  print('Loading trees ...')
  
  rndstate = random.getstate()
  random.seed(0)
  size=np.minimum(len(sig),len(bkg))
  print('sg length=',len(sig))
  
  train_frac=train_frac_rel
  val_frac=train_frac+val_frac_rel
  test_frac=val_frac+test_frac_rel

  N_train=int(train_frac*size)
  Nval=int(val_frac*size)
  Ntest=int(test_frac*size)

  train=sig[0:N_train]+bkg[0:N_train]
  dev=sig[N_train:Nval]+bkg[N_train:Nval]
  test=sig[Nval:Ntest]+bkg[Nval:Ntest]

  random.shuffle(train)
  random.shuffle(dev)
  random.shuffle(test)
  random.setstate(rndstate)
  
  train=np.asarray(train)
  dev=np.asarray(dev)
  test=np.asarray(test)
  
  print('Train shape=',train.shape)
  #We reshape for single jets studies (Modify the code for full events)
#   train=train.reshape(train.shape[0]*train.shape[1])
#   dev=dev.reshape(dev.shape[0]*dev.shape[1])
# #   print(test.shape)
#   test=test.reshape(test.shape[0]*test.shape[1]) 



  print('Size data each sg and bg =',size)
  print('Length train =', len(train))

  return train, dev, test
  
  
class RecursiveNetStaticGraph():

  def __init__(self, config):
    self.config = config


    sig_tree, sig_list=tree.makeTrees(dir_jets_subjets,sg,myN_jets,0)
    bkg_tree, bkg_list=tree.makeTrees(dir_jets_subjets,bg,myN_jets,0)
    
    
    # Load train data and build vocabulary
#     self.train_list, self.dev_list, self.test_list = split_sample(sig_tree, bkg_tree, 0.6, 0.2, 0.2)
    self.train_data, self.dev_data, self.test_data = split_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)
    
  
  def read_data(self, events):
    events_contents=[]
    events_left=[]
    events_right=[]
    events_labels=[]
    for event in events:
      contents=[jet['content'] for jet in event][0] #Change for full event studies
      left=[jet['left'] for jet in event][0]
      right=[jet['right'] for jet in event][0]
      labels=[jet['label'] for jet in event]
    
      events_contents.append(contents)
      events_left.append(left)
      events_right.append(right)
      events_labels.append(labels)
    
    return np.asarray(events_contents), np.asarray(events_left), np.asarray(events_right), np.asarray(events_labels) 
    
   

  def batch_data(self, train_contents, train_labels):
    # Generate BATCH_SIZE sample pairs
    contents_batch, labels_batch= tf.train.batch(
    [train_contents, train_labels], 
    batch_size=self.config.batch_size, 
    num_threads=10, 
    capacity=self.config.capacity, 
    enqueue_many=True, 
    shapes=None, 
    dynamic_pad=True, 
    allow_smaller_final_batch=True
)
    return contents_batch, labels_batch    
    
# def test_RNN():
  """Test RNN model implementation.
  """
config = Config()
model = RecursiveNetStaticGraph(config)
  #graph_def = tf.get_default_graph().as_graph_def()
  #with open('static_graph.pb', 'wb') as f:
  #  f.write(graph_def.SerializeToString())

# //////////////////////////////
train_contents, train_left, train_right, train_labels = model.read_data(model.train_data)
#   contents_batch, labels_batch= model.batch_data(train_contents, train_labels)
#   contents_batch, left_batch, right_batch, labels_batch= model.batch_data(train_contents, train_left, train_right, train_labels)
#   model.generate_batches(contents_batch, left_batch, right_batch, labels_batch)

# //    
    
    
    
  
  
# dir = os.path.dirname(os.path.realpath(__file__))

# We simulates some raw inputs data
# let's say we receive 100 batches, each containing 50 elements
# x_inputs_data = tf.random_normal([2], mean=0, stddev=1)
# q = tf.FIFOQueue(capacity=10, dtypes=tf.float32)
# enqueue_op = q.enqueue_many(x_inputs_data)

# input = q.dequeue()

# numberOfThreads = 1
# qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)

batch_input = tf.train.batch(
    [train_contents], 
    batch_size=2, 
    num_threads=5, 
    capacity=32, 
    enqueue_many=False, 
    shapes=None, 
    dynamic_pad=False, 
    allow_smaller_final_batch=False
)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # threads = qr.create_threads(sess, coord=coord, start=True)

    print(sess.run([batch_input]))
    coord.request_stop()
    coord.join(threads)