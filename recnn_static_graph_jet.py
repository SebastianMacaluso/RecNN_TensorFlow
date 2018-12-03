##---------------------------------------------------------------------------------------------
# This code works well. It loads tree_list and then rearranges the tree contents before loading the data into the placeholders. We do training jet by jet, so we will add batching in the future. (Also in "tree_batch" we rearrange the tree contents before loading the trees)
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
import tree_list as tree 

np.set_printoptions(threshold=np.nan)

import scipy 

import time
start_time = time.time()

import image_preprocess_functions_color as fns

random.seed(0)




# n = 0
# x = tf.constant(list(range(n)))
# c = lambda i, x: n < i
# b = lambda i, x: (tf.Print(i - 1, [i]), tf.Print(x + 1, [i], "x:"))
# i, out = tf.while_loop(c, b, (10, x))
# with tf.Session() as sess:
#     print(sess.run(i))  # prints [0] ... [9999]
# 
#     # The following line may increment the counter and x in parallel.
#     # The counter thread may get ahead of the other thread, but not the
#     # other way around. So you may see things like
#     # [9996] x:[9987]
#     # meaning that the counter thread is on iteration 9996,
#     # while the other thread is on iteration 9987
#     print(sess.run(out).shape)
# 
# sys.exit()


##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

# myN_jets=1000000000
myN_jets=50

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


#----------------------------------------------------------------------
class RecursiveNetStaticGraph():

  def __init__(self, config):
    self.config = config


    sig_tree, sig_list=tree.makeTrees(dir_jets_subjets,sg,myN_jets,0)
    bkg_tree, bkg_list=tree.makeTrees(dir_jets_subjets,bg,myN_jets,0)
    
    
    # Load train data and build vocabulary
#     self.train_list, self.dev_list, self.test_list = split_sample(sig_tree, bkg_tree, 0.6, 0.2, 0.2)
    self.train_data, self.dev_data, self.test_data = split_sample(sig_list, bkg_list, 0.6, 0.2, 0.2)
    
    
    
#     def build_feed_dict(self, jets):
#   
#       contents=[jet['content'] for jet in jets][0] #Change for full event studies
# #       print('contents=',contents)
#       print('---'*20)
#       trees=[jet['tree'] for jet in jets][0]#Change for full event studies
#       labels=[jet['label'] for jet in jets]
#       
#       
#     
#       contents=contents[::-1]
#       d=len(contents)
#       left=[-1 if tree[0]==-1 else (d-1-tree[0]) for tree in trees]
#       right=[-1 if tree[0]==-1 else (d-1-tree[1]) for tree in trees]
#       
#       left=left[::-1]
#       right=right[::-1]
#       
#       
#       print('Content=',contents)
#       print('Length contents=',len(contents))
#       print('Trees=',trees)
#       print('Left children=',left)
#       print('Right children=',right)
#       
#       # Below we check that the tree contains the right location of each children subjet 
#       ii = 9;
# #       print('Content = ',content[0])
#       print('Content ',ii,' = ',contents[ii])
#       print('Left Children location =',left[ii])
#       print('Content ',ii,' by adding the 2 children 4-vectors= ',contents[left[ii]] 
#       + contents[right[ii]])
#       
#     build_feed_dict(self, self.train_data[0])
#     
#     sys.exit()
    
#     print('self.train_data =',self.train_data[0])
    
    
#     sys.exit()
#     print('Train data=',self.train_data)
    
#     self.vocab = utils.Vocab()
#     train_sents=[]
#     for t in self.train_data:
#       train_sents.append(t.get_words())
    # train_sents = [t.get_words() for t in self.train_data] #We get the 4-vectors of the leaves
#     train_sents = [np.reshape(t.get_words(),(len(t.get_words()),4)) for t in self.train_data] #We get the 4-vectors of the leaves
#     self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))
#     train_sents=np.asarray(train_sents)
#     train_sents= np.reshape(train_sents,(len(train_sents),-1,4))
#     train_sents[0]= np.reshape(train_sents[0],(len(train_sents[0]),4))
#     print('Type train_sents=', train_sents[0][0].dtype)
#     print('Train_sents=',train_sents[0][0])
#     mytype= [t.root.right.type for t in self.train_data]
#     print('Type=',mytype)
#     sys.exit()

# --------------------
# PLACEHOLDERS
    # add input placeholders
    self.content_placeholder = tf.placeholder(tf.float32,(None), name='content_placeholder')
    self.trees_placeholder = tf.placeholder(tf.int32,(None), name='trees_placeholder')
    self.labels_placeholder = tf.placeholder(tf.int32,(None), name='labels_placeholder')
    

#     self.vec_placeholder = tf.placeholder(tf.float32,(None), name='vec_placeholder')
    self.is_leaf_placeholder = tf.placeholder(tf.bool, (None), name='is_leaf_placeholder')
    self.left_children_placeholder = tf.placeholder(
        tf.int32, (None), name='left_children_placeholder')
    self.right_children_placeholder = tf.placeholder(
        tf.int32, (None), name='right_children_placeholder')
# #     self.node_word_indices_placeholder = tf.placeholder(tf.float64, [None, 4], name='node_word_indices_placeholder')
#     self.node_word_indices_placeholder = tf.placeholder(tf.int32, (None), name='node_word_indices_placeholder')
# #     self.labels_placeholder = tf.placeholder(tf.int32, [None, 2], name='labels_placeholder')
#     self.labels_placeholder = tf.placeholder(tf.int32, (None), name='labels_placeholder')
# -----------------------
# -----------------------
# MODEL VARIABLES
    # add model variables
    with tf.variable_scope('Embed'):
      Wu = tf.get_variable('Wu',
                                   [4,self.config.embed_size]) #Added the 4 for the 4-vector 
      bu = tf.get_variable('bu', [1, self.config.embed_size])
    #----
#     with tf.variable_scope('Embeddings'):
#       embeddings = tf.get_variable('embeddings',
#                                    [len(train_sents), self.config.embed_size,4]) #Added the 4 for the 4-vector dimension
    with tf.variable_scope('Composition'):
      W1 = tf.get_variable('W1',
                           [3 * self.config.embed_size, self.config.embed_size])
      b1 = tf.get_variable('b1', [1, self.config.embed_size])
    with tf.variable_scope('Projection'):
      U = tf.get_variable('U', [self.config.embed_size, self.config.label_size]) # label_size gives the number of classes for classification (2 in our case= sg or bg)
      bs = tf.get_variable('bs', [1, self.config.label_size])

    # build recursive graph

    tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)




#   
#     def rec_embedding(content,tree, index, args=None):
#    
#       """
#       Recursive function traverses tree
#       from left to right. 
#       Calls nodeFn at each node
#       """
#       index=tf.to_int32(index)
# #       index=int(index)
#   #       print('content[index]=',content[index])
#     
#     
#   #         vect=np.reshape(content[index],(4,1))
#   #         print('vec shape=',vect.shape)
#       # vect=np.float32(content[index])
#       vect=content[index]
#   #         u_k=
#   #         print('Shape u_k',u_k.shape)
#   #         print('Is leaf?=',node.isLeaf)
#       def leaf_conv(vect):
#         u_k=tf.nn.relu(tf.matmul(tf.transpose(vect), Wu) + bu)
#         return u_k
#       
#       def inner_emb(content,tree, index):
#         h_L=rec_embedding(content,tree, tensor_array.read(tree[index,0]))
#         h_R=rec_embedding(content,tree,tensor_array.read(tree[index,1]))
#         #           print('Shape h_L=',h_L.shape)
#         h=tf.nn.relu(tf.matmul(tf.concat([h_L, h_R,leaf_conv(vect)],1), W1) + b1)
#         return h
#     
#       tf.cond(tree[index,0]<0,
#               lambda: leaf_conv(vect),
#               lambda: inner_emb(content,tree,index)
#               )
              
              
    def leaf_conv(vect):
      with tf.device(self.config.mygpu):
        u_k=tf.nn.relu(tf.matmul(tf.transpose(vect), Wu) + bu)
        return u_k
          
    def inner_conv(vect,left_tensor, right_tensor):
      with tf.device(self.config.mygpu):
        h=tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor,leaf_conv(vect)],1), W1) + b1)
        return h 
  
  #-------------------------------------------   
    p=tf.to_int32(tf.shape(self.content_placeholder)[0]-1) 
    
    def loop_body(tensor_array, i):
      node_is_leaf = tf.gather(self.is_leaf_placeholder, i) #tf.gather(param, indices) It gathers slices from params according to indices.
#       node_word_index = tf.gather(self.node_word_indices_placeholder, i)
      left_child = tf.gather(self.left_children_placeholder, i)
      right_child = tf.gather(self.right_children_placeholder, i)
#       
#       four_vector=tf.gather(self.vec_placeholder,i)
      content=tf.gather(self.content_placeholder,i)
#       tree=tf.gather(self.trees_placeholder,i)
      
      print('tf.shape(self.content_placeholder)=',tf.shape(self.content_placeholder))
      
      node_tensor = tf.cond(
          node_is_leaf,
          lambda: leaf_conv(content),
          lambda: inner_conv(content,tensor_array.read(left_child),
                                   tensor_array.read(right_child))
                                   )
      print('node_tensor=',node_tensor) 
#       print('--'*20)                             
   
#     def loop_body(tensor_array, i):
#       node_is_leaf = tf.gather(self.is_leaf_placeholder, i)
#       node_word_index = tf.gather(self.node_word_indices_placeholder, i)
#       left_child = tf.gather(self.left_children_placeholder, i)
#       right_child = tf.gather(self.right_children_placeholder, i)
#       node_tensor = tf.cond(
#           node_is_leaf,
#           lambda: embed_word(node_word_index),
#           lambda: combine_children(tensor_array.read(left_child),
#                                    tensor_array.read(right_child)))                       
                                   
                                   
      tensor_array = tensor_array.write(i, node_tensor) #We write this new graph 
      
      i = tf.add(i, 1) #We go to the next leaf
#       print('current i=',tf.to_int32(i))
#       print('----'*20)
      return tensor_array,i

    loop_cond = lambda tensor_array, i: \
        tf.less(i, tf.squeeze(tf.shape(self.is_leaf_placeholder))) #Returns the truth value of (x < y) element-wise. (tf.squeeze Removes dimensions of size 1 from the shape of a tensor. ). We do this so that i stops when we loop over all the elements


# 
# # ========================================
#     def loop_body(tensor_array, i):
#       node_is_leaf = tf.gather(self.is_leaf_placeholder, i) #tf.gather(param, indices) It gathers slices from params according to indices.
# #       node_word_index = tf.gather(self.node_word_indices_placeholder, i)
#       left_child = tf.gather(self.left_children_placeholder, i)
#       right_child = tf.gather(self.right_children_placeholder, i)
# #       
# #       four_vector=tf.gather(self.vec_placeholder,i)
#       content=tf.gather(self.content_placeholder,i)
#       tree=tf.gather(self.trees_placeholder,i)
#       
#       
#       
#       node_tensor = tf.cond(
#           node_is_leaf,
#           lambda: leaf_conv(content),
#           lambda: inner_conv(content,tensor_array.read(left_child),
#                                    tensor_array.read(right_child)))
#       print('node_tensor=',node_tensor) 
#       print('--'*20)                             
#                           
#                                    
#                                    
#       tensor_array = tensor_array.write(i, node_tensor) #We write this new graph 
#       
#       i = tf.add(i, 1) #We go to the next leaf
#       print('current i=',tf.to_int32(i))
#       print('----'*20)
#       return tensor_array, i
#       
# ================================

# HERE WE IMPLEMENT THE GRAPH. Repeat body while the condition cond is true.
#     i_start=tf.to_int32(tf.shape(self.content_placeholder)[0]-1)
    i_start=0
#     print('i_start=',i_start)
    self.tensor_array, _ = tf.while_loop(
        loop_cond, loop_body, [tensor_array, i_start], parallel_iterations=120)







    # add projection layer
    with tf.device(self.config.mygpu):
      self.logits = tf.matmul(self.tensor_array.concat(), U) + bs
      self.root_logits = tf.matmul(
          self.tensor_array.read(self.tensor_array.size() - 1), U) + bs
      self.root_prediction = tf.squeeze(tf.argmax(self.root_logits, 1)) #We find the class with the max probability

    # add loss layer
    regularization_loss = self.config.l2 * (
        tf.nn.l2_loss(W1) + tf.nn.l2_loss(U))
    included_indices = tf.where(tf.less(self.labels_placeholder, 2))
    self.full_loss = regularization_loss + tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits =tf.gather(self.logits, included_indices), labels = tf.gather(
                self.labels_placeholder, included_indices)))
                
    self.root_loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits =self.root_logits, labels=self.labels_placeholder[-1:]))
#     self.full_loss = regularization_loss + tf.reduce_sum(
#         tf.nn.sparse_softmax_cross_entropy_with_logits(
#             tf.gather(self.logits, included_indices), tf.gather(
#                 self.labels_placeholder, included_indices)))
#     self.root_loss = tf.reduce_sum(
#         tf.nn.sparse_softmax_cross_entropy_with_logits(
#             self.root_logits, self.labels_placeholder[-1:]))

    # add training op
    self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
        self.full_loss)


#---------------------
  def build_feed_dict(self, jets):
    contents=[jet['content'] for jet in jets][0] #Change for full event studies
#       print('contents=',contents)
#     print('---'*20)
    tree=[jet['tree'] for jet in jets][0]#Change for full event studies
    labels=[jet['label'] for jet in jets]
    
#     labels=np.asarray(labels)
#     print('labels shape=',labels.shape)
  
    contents=contents[::-1]
    d=len(contents)
    left=[-1 if node[0]==-1 else (d-1-node[0]) for node in tree]
    right=[-1 if node[1]==-1 else (d-1-node[1]) for node in tree]
    
    left=np.asarray(left[::-1])
    right=np.asarray(right[::-1])
    
    
#     print('Content=',contents)
#     print('Length contents=',len(contents))
#     print('Tree=',tree)
#     print('Left children=',left)
#     print('Right children=',right)
#     
#     # Below we check that the tree contains the right location of each children subjet 
#     ii = 9;
# #       print('Content = ',content[0])
#     print('Content ',ii,' = ',contents[ii])
#     print('Left Children location =',left[ii])
#     print('Content ',ii,' by adding the 2 children 4-vectors= ',contents[left[ii]] 
#     + contents[right[ii]])

  
#     print('Trees:',trees)
#     print('Tree entries:',[tree for tree in trees])
#     print('is_leaf=',[True if tree[0]==-1 else False for tree in trees])
#     
#     nodes_list = []
#     tree.leftTraverse(node, lambda node, args: args.append(node), nodes_list) #We traverse the whole tree and apply the lambda function in each node. So we append all the nodes to nodes_list
#     node_to_index = OrderedDict()
#     for i in range(len(nodes_list)):
#       node_to_index[nodes_list[i]] = i #We add the nodes to the OrderedDict where the keys are the nodes and the values the index in the list. This maps each node to an index
#     vector_list=[node.word for node in nodes_list]
#     vector_list=np.reshape(vector_list,(len(vector_list),4,1))
#     vector_list=np.float32(vector_list)

    feed_dict = {
    
#     
#         #We create a list with True/False if the node is a leaf or not. Type bool
        self.is_leaf_placeholder: [True if node==-1 else False for node in left],
#         #We create a list for the index in the nodes_list of the left child of each node. If it is a leaf we get a -1. Type int32 
#         self.left_children_placeholder: [tree[0] for tree in trees],
#         self.right_children_placeholder: [tree[1] for tree in trees],
        self.left_children_placeholder: left,
        self.right_children_placeholder: right,
        
        
#         # We get a list for the encoding of each word in our vocabulary. In our case these should be the 4-vectors and should be type float64 (and dim=1x4) ?     
#         self.vec_placeholder: vector_list,                                  
# #         self.node_word_indices_placeholder: [node.word for node in nodes_list],
# #         self.node_word_indices_placeholder: [self.vocab.encode(node.word) if
# #                                              node.word else -1
# #                                              for node in nodes_list],
        self.content_placeholder: contents,
#         self.trees_placeholder: tree,
        self.labels_placeholder: labels
#         self.labels_placeholder: [node.type for node in nodes_list] ############# Change labels for type, but add a type to all the nodes that are not the root
    }
    
#     print('feed dict=',feed_dict)
#     print('feed_dict[vec_placeholder] type=',feed_dict[self.vec_placeholder].dtype)
#     print('---'*20)


#     print('self.is_leaf_placeholder shape=',np.asarray(feed_dict[self.is_leaf_placeholder]).shape)
#     print('self.is_leaf_placeholder =',feed_dict[self.is_leaf_placeholder])
#     print('self.left_children_placeholder =',feed_dict[self.left_children_placeholder])
#     print('self.right_children_placeholder =',feed_dict[self.right_children_placeholder])
#     print('self.content_placeholder =',feed_dict[self.content_placeholder])
#     print('self.labels_placeholder =',feed_dict[self.labels_placeholder])
    return feed_dict

  def predict(self, trees, weights_path, get_loss=False,args=None):
    """Make predictions from the provided model."""
    results = []
    losses = []
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, weights_path)
      for tree in trees:
        # feed_dict = self.build_feed_dict(tree.root)
        feed_dict = self.build_feed_dict(tree)
        if get_loss:
          root_prediction, loss = sess.run(
              [self.root_prediction, self.root_loss], feed_dict=feed_dict)
          losses.append(loss)
        else:
          root_prediction = sess.run(self.root_prediction, feed_dict=feed_dict)
        results.append(root_prediction)
    return results, losses

  def run_epoch(self, new_model=False, verbose=True):
    loss_history = []
    # training
    random.shuffle(self.train_data)
#     random.shuffle(self.train_list)
    with tf.Session() as sess:
      if new_model:
        sess.run(tf.global_variables_initializer()) 
        # sess.run(tf.initialize_all_variables())
      else:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
      for step, tree in enumerate(self.train_data):
        feed_dict = self.build_feed_dict(tree)
        loss_value, _ = sess.run([self.full_loss, self.train_op],
                                 feed_dict=feed_dict)
        loss_history.append(loss_value)
        if verbose:
          sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
              self.train_data), np.mean(loss_history)))
          sys.stdout.flush()
      saver = tf.train.Saver()
      if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
      saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
    # statistics
    train_preds, _ = self.predict(self.train_data,
                                  SAVE_DIR + '%s.temp' % self.config.model_name)
    val_preds, val_losses = self.predict(
        self.dev_data,
        SAVE_DIR + '%s.temp' % self.config.model_name,
        get_loss=True)
    train_labels=[jet[0]['label'] for jet in self.train_data]
    val_labels=[jet[0]['label'] for jet in self.dev_data]
    
 #    train_labels = [t.root.type for t in self.train_data] #Changed label for type (in the old code)
#     val_labels = [t.root.type for t in self.dev_data] #Changed label for type (in the old code)
    train_acc = np.equal(train_preds, train_labels).mean()
    val_acc = np.equal(val_preds, val_labels).mean()

    
    print('Training acc (only root node): {}'.format(train_acc))
    print('Valiation acc (only root node): {}'.format(val_acc))
    print(self.make_conf(train_labels, train_preds))
    print(self.make_conf(val_labels, val_preds))
    return train_acc, val_acc, loss_history, np.mean(val_losses)

  def train(self, verbose=True):
    complete_loss_history = []
    train_acc_history = []
    val_acc_history = []
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_epoch = 0
    stopped = -1
    for epoch in range(self.config.max_epochs):
      print('epoch %d' % epoch)
      if epoch == 0:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch(
            new_model=True)
      else:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch()
      complete_loss_history.extend(loss_history)
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      #lr annealing
      epoch_loss = np.mean(loss_history)
      if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
        self.config.lr /= self.config.anneal_by
        print('annealed lr to %f' % self.config.lr)
      prev_epoch_loss = epoch_loss
      
# 
#       #save if model has improved on val
#       if val_loss < best_val_loss:
#         shutil.copyfile(SAVE_DIR + '%s.temp' % self.config.model_name,
#                         SAVE_DIR + '%s' % self.config.model_name)
#         best_val_loss = val_loss
#         best_val_epoch = epoch

      # if model has not imprvoved for a while stop
      if epoch - best_val_epoch > self.config.early_stopping:
        stopped = epoch
        #break
    if verbose:
      sys.stdout.write('\r')
      sys.stdout.flush()

    print('\n\nstopped at %d\n' % stopped)
    return {
        'loss_history': complete_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

  def make_conf(self, labels, predictions):
    confmat = np.zeros([2, 2])
    for l, p in zip(labels, predictions):
      confmat[l, p] += 1
    return confmat

#################################################
def plot_loss_history(stats):
  plt.plot(stats['loss_history'])
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.savefig('loss_history.png')
#   plt.show()


def test_RNN():
  """Test RNN model implementation.
  """
  config = Config()
  model = RecursiveNetStaticGraph(config)
  #graph_def = tf.get_default_graph().as_graph_def()
  #with open('static_graph.pb', 'wb') as f:
  #  f.write(graph_def.SerializeToString())

  start_time = time.time()
  stats = model.train(verbose=True)
  print('Training time: {}'.format(time.time() - start_time))

  plot_loss_history(stats)

  start_time = time.time()
  val_preds, val_losses = model.predict(
      model.dev_data,
      SAVE_DIR + '%s.temp' % model.config.model_name,
      get_loss=True)
      
  val_labels=[jet[0]['label'] for jet in model.dev_data]
#   val_labels = [t.root.type for t in model.dev_data] #Changed label for type (in the old code)
  val_acc = np.equal(val_preds, val_labels).mean()
  print(val_acc)

  print( '-' * 20)
  print ('Test')
  predictions, _ = model.predict(model.test_data,
                                 SAVE_DIR + '%s.temp' % model.config.model_name)
  labels=[jet[0]['label'] for jet in model.test_data]                               
#   labels = [t.root.type for t in model.test_data] #Changed label for type (in the old code)
  print(model.make_conf(labels, predictions))
  test_acc = np.equal(predictions, labels).mean()
  print ('Test acc: {}'.format(test_acc))
  print ('Inference time, dev+test: {}'.format(time.time() - start_time))
  print ('-' * 20)

##-----------------------------------------------
if __name__ == '__main__':
  
#   print('Loading signal ...')
#   sig=tree.makeTrees(dir_jets_subjets,sg,myN_jets,0)
#   print('Loading background ...')
#   bkg=tree.makeTrees(dir_jets_subjets,bg,myN_jets,0)
#   split_sample(sig, bkg, 0.6, 0.2, 0.2)
  test_RNN()
