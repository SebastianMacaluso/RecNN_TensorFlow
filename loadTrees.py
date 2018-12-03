


#To run this script:
# python loadTrees.py
##---------------------------------------------------------------------------------------------
#RESOLUTION of ATLAS/CMS

# CMS ECal DeltaR=0.0175 and HCal DeltaR=0.0875 (https://link.springer.com/article/10.1007/s12043-007-0229-8 and https://cds.cern.ch/record/357153/files/CMS_HCAL_TDR.pdf ) 
# CMS: For the endcap region, the total number of depths is not as tightly constrained as in the barrel due to the decreased φ-segmentation from 5 degrees (0.087 rad) to 10 degrees for 1.74 < |η| < 3.0. (http://inspirehep.net/record/1193237/files/CMS-TDR-010.pdf)
#The endcap hadron calorimeter (HE) covers a rapidity region between 1.3 and 3.0 with good hermiticity, good
# transverse granularity, moderate energy resolution and a sufficient depth. A lateral granularity ( x ) was chosen
# 0.087 x 0.087. The hadron calorimeter granularity must match the EM granularity to simplify the trigger. (https://cds.cern.ch/record/357153/files/CMS_HCAL_TDR.pdf )

# ATLAS ECal DeltaR=0.025 and HCal DeltaR=0.1 (https://arxiv.org/pdf/hep-ph/9703204.pdf page 11)

##---------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------
#LOAD LIBRARIES
from __future__ import print_function
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


import tree_class

np.set_printoptions(threshold=np.nan)

import scipy 

import time
start_time = time.time()

import image_preprocess_functions_color as fns

random.seed(0)

##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

myN_jets=3



dir_jets_subjets='inputTrees' #+batch_label




# dir_jets_subjets='../pyroot/Output_'+dir_label+'/' #+batch_label
std_dir='standardization/'

os.system('mkdir -p image_array')
os.system('mkdir -p plots')
Images_dir=local_dir+'plots/' #Output dir to save the image plots
image_array_dir=local_dir+'image_array/' #Output dir to save the image arrays


###############################################################################
# FUNCTIONS
###############################################################################


# string should be either qcd or tt
def preprocess(dir_subjets,string,N_jets,truthval):

  print('Loading files for subjets')
#   print('Subjet array format ([[[pTsubj1],[pTsubj2],...],[[etasubj1],[etasubj2],...],[[phisubj1],[phisubj2],...]])')
  print('-----------'*10)
  

  subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if ('tree' in filename and string in filename and filename.endswith('.dat'))]
  N_analysis=len(subjetlist)
  print('Number of subjet files =',N_analysis)
  print('Loading subjet files...  \n {}'.format(subjetlist))
  print('Subjets list',subjetlist)
#   images=[]
#   jetmasslist=[]

  Ntotjets=0


    
  for ifile in range(N_analysis):
#    print(myN_jets,Ntotjets)

#     if(Ntotjets>N_jets):
#       print('Leaving....')
#       break
       
    for s in open(dir_subjets+'/'+subjetlist[ifile]):
  #        print(s)
      if(Ntotjets>N_jets):
        print('Leaving....')
        sys.exit()
     
      event=json.loads(s)
  #        print('Full event tree = ',event[0])
      Ntotjets+=1
      print('Ntotjets = ', Ntotjets)
    

    
      tree=np.asarray(event[0])
      content=np.asarray(event[1])
      mass=np.asarray(event[2])
      pt=np.asarray(event[3])
    
      tree=np.array([np.asarray(e).reshape(-1,2) for e in tree])
      content=np.array([np.asarray(e).reshape(-1,4) for e in content])
    
      print('tree = ',tree[0])
      print('content = ',content[0])
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
#       def builTree(content,tree):
      for i in range(len(content)):
        tree_class.Tree(content[i],tree[i])
      
        print('Leaves = ',tree_class.Tree.get_words)

      
      
      
      
      
      
      
      
      
      
      
      
      jets=[]
      for i in range(len(tree)):
    
        jet = {}
 
        jet["root_id"] = 0
        jet["tree"] = tree[i] #Labels for the jet constituents in the tree (SM)
        jet["content"] = content[i] #Where content[i][0] is the jet 4-momentum, and the other entries are the jets constituents 4 momentum? (SM)
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
        
        jets.append(jet)
        
        print('Length jets =', len(jets))
     





    
##-----------------------------------------------------
if __name__=='__main__':

    preprocess(dir_jets_subjets,'tt',myN_jets,0)
    
#     builTree(content,tree)

    print('FINISHED.')

    print('-----------'*10)
    print("Code execution time = %s minutes" % ((time.time() - start_time)/60))
    print('-----------'*10) 
  


  
  
  
