
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

from data_ops import batching_tf as batching
##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

# myN_jets=1000000000
myN_jets=2



dir_jets_subjets='inputTrees' #+batch_label

sg='tt'


# dir_jets_subjets='../pyroot/Output_'+dir_label+'/' #+batch_label
std_dir='standardization/'

os.system('mkdir -p image_array')
os.system('mkdir -p plots')
Images_dir=local_dir+'plots/' #Output dir to save the image plots
image_array_dir=local_dir+'image_array/' #Output dir to save the image arrays

###############################################################################
# CLASSES
###############################################################################
class Node:  # a node in the tree
    def __init__(self, id, word=None):
        
        self.type = None
        self.label = id
          
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)
        
        

class Tree:

    def __init__(self, tokens, treelabels):
#         tokens = []
#         self.open = '('
#         self.close = ')'
#         for toks in treeString.strip().split(): #This is a line in a tex file - So they are all the leaves of the tree
#             tokens += list(toks) #We list all the words in the file -- All the leaves of the tree
        self.root = self.parse(tokens,treelabels, 0) #element 0 is the root (the jet in our case)
        
#         
#         # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, treelabels, index, parent=None):

        index=int(index)
#         print('index = ',index)
#         print('4-vector =', tokens[index])
#         print('----------------'*10)
        node = Node(index)  # zero index labels
        node.word=tokens[index]   
        node.parent = parent

        if treelabels[index,0]<0: #It is a leaf
          node.isLeaf=True
        else:
          # We set the left and right children
          node.left = self.parse(tokens, treelabels, treelabels[index,0], parent=node)
          node.right = self.parse(tokens, treelabels, treelabels[index,1], parent=node)

# 
# 
#       # Below we check that the tree contains the right location of each children subjet 
#       ii = 3;
# #       print('Content = ',content[0])
#       print('Content ',ii,' = ',content[0][ii])
#       print('Children location =',tree[0][ii])
#       print('Content ',ii,' by adding the 2 children 4-vectors= ',content[0][tree[0][ii,0]] 
#       + content[0][tree[0][ii,1]])

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)
        
        
def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]
    
def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)



def simplified_data(binTree, type, num_train, num_dev, num_test):
    rndstate = random.getstate()
    random.seed(0)
    
    binarize_labels(type,binTree)
#     trees = binTree
    
#     We label sg and bg trees
    for t in binTree:    
      if str(type)=='sg':
        t.root.type = 1
      else:
        t.root.type = 0
        
        
#     
#     #filter extreme trees
#     pos_trees = [t for t in trees if t.root.label==4]
#     neg_trees = [t for t in trees if t.root.label==0]
# 
#     #binarize labels
#     binarize_labels(pos_trees)
#     binarize_labels(neg_trees)
    
    #split into train, dev, test
#     print(len(pos_trees), len(neg_trees))
    pos_trees = sorted(pos_trees, key=lambda t: len(t.get_words()))
    neg_trees = sorted(neg_trees, key=lambda t: len(t.get_words()))
    num_train/=2
    num_dev/=2
    num_test/=2
    train = pos_trees[:num_train] + neg_trees[:num_train]
    dev = pos_trees[num_train : num_train+num_dev] + neg_trees[num_train : num_train+num_dev]
    test = pos_trees[num_train+num_dev : num_train+num_dev+num_test] + neg_trees[num_train+num_dev : num_train+num_dev+num_test]
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    random.setstate(rndstate)


    return train, dev, test


def binarize_labels(type, trees):
    def binarize_node(node, _):
#         if node.label<2:
#             node.label = 0
#         elif node.label>2:
#             node.label = 1
#             for t in binTree:    
      if str(type)==sg:
        node.type = 1
      else:
        node.type = 0
    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)
        
        
####################################################################
# string should be either qcd or tt
def makeTrees(dir_subjets,string,N_jets,truthval):

#   print('Loading files for subjets')
#   print('Subjet array format ([[[pTsubj1],[pTsubj2],...],[[etasubj1],[etasubj2],...],[[phisubj1],[phisubj2],...]])')
#   print('-----------'*10)
  

  subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if ('tree' in filename and string in filename and filename.endswith('.dat'))]
  N_analysis=len(subjetlist)
  print('Number of subjet files =',N_analysis)
  print('Loading subjet files...  \n {}'.format(subjetlist))
  print('Subjets list',subjetlist)
#   images=[]
#   jetmasslist=[]

  Ntotjets=0


  final_trees=[]  
  jets=[]
  for ifile in range(N_analysis):
#    print(myN_jets,Ntotjets)

#     if(Ntotjets>N_jets):
#       print('Leaving....')
#       break
       
    for s in open(dir_subjets+'/'+subjetlist[ifile]):
  #        print(s)
#       if(Ntotjets>N_jets):
#         print('Leaving....')
#         sys.exit()

      if(Ntotjets<N_jets):
        event=json.loads(s)
    #        print('Full event tree = ',event[0])
        Ntotjets+=1
#         print('Ntotjets = ', Ntotjets)
    

    
        tree=np.asarray(event[0])
        content=np.asarray(event[1])
        mass=np.asarray(event[2])
        pt=np.asarray(event[3])
    
        tree=np.array([np.asarray(e).reshape(-1,2) for e in tree])
        content=np.array([np.asarray(e).reshape(-1,4) for e in content])
    
  #       print('tree = ',tree[0])
  #       print('content = ',content[0])
  #       print('mass =',mass)
  #       print('pt = ',pt)


        # Below we check that the tree contains the right location of each children subjet 
        ii = 3;
  #       print('Content = ',content[0])
        print('Content ',ii,' = ',content[0][ii])
        print('Children location =',tree[0][ii])
        print('Content ',ii,' by adding the 2 children 4-vectors= ',content[0][tree[0][ii,0]] 
        + content[0][tree[0][ii,1]])

        print('-------------------'*10)
      
  #       Create the trees and nodes

#         temp_trees = [Tree(content[i],tree[i]) for i in range(len(content))]
        temp_trees = [Tree(content[i],tree[i]) for i in range(1)] #This way we get only 1 tree per event. This should be modified for full events studies where we have more than 1 jet (In this case replace for the previous line and update recnn_static_graph.py)
      
        binarize_labels(string,temp_trees)

        final_trees.append(temp_trees) #Uncomment for full event studies
#   print('Final trees =', final_trees)

          
        event=[]
        # for i in range(len(tree)):
        for i in range(1): #This only works for single jet studies. Modify for full events
        
          

          contents=np.reshape(content[i],(-1,4,1)) #Where content[i][0] is the jet 4-momentum, and the other entries are the jets constituents 4 momentum? (SM)
          contents=contents[::-1]
          
          d=len(contents)
          jet_tree=tree[i] #Labels for the jet constituents in the tree (SM)
          left=[-1 if node[0]==-1 else (d-1-node[0]) for node in jet_tree]
          right=[-1 if node[1]==-1 else (d-1-node[1]) for node in jet_tree]
    
          left=np.asarray(left[::-1])
          right=np.asarray(right[::-1])
          
          
          jet = {}
   
          jet["root_id"] = 0
#           jet["tree"] = 
          jet["content"] = contents
          jet["left"]=left
          jet["right"]=right
          jet["mass"] = mass[i]
          jet["pt"] = pt[i]
          jet["energy"] = content[i][0, 3]
  
          px = content[i][0, 0] #The jet is the first entry of content. And then we have (px,py,pz,E)
          py = content[i][0, 1]
          pz = content[i][0, 2]
          p = (content[i][0, 0:3] ** 2).sum() ** 0.5
  #         jet["Calc energy"]=(p**2+mass[i]**2)**0.5
          eta = 0.5 * (np.log(p + pz) - np.log(p - pz)) #pseudorapidity eta
          phi = np.arctan2(py, px)
   
          jet["eta"] = eta
          jet["phi"] = phi
  #         print('jet contents =', jet.items())
          if str(string)==sg:
            jet["label"]=1
          else:
            jet["label"]=0
            
          jets.append(jet)  
#           event.append(jet) #Uncomment for full event studies
        
#         jets.append(event)  #Uncomment for full event studies
          print('Length jets =', len(jets))

  print('Number of trees =', len(final_trees))
      
      
  return final_trees, jets
     
# ===========================================================================  
if __name__=='__main__':
  sig_tree, sig_list=makeTrees(dir_jets_subjets,sg,myN_jets,0)
  levels, children, n_inners, contents= batching.batch(sig_list)
  
  
  
  
  
  