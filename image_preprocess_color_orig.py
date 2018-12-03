# -*- coding: utf-8 -*-
# This script loads .npy files as a list of numpy arrays ([[pT],[eta],[phi]]) and produces numpy arrays where each entry represents the intensity in pT for a pixel in a jet image. The script does the following:
# 1) We load .npy files with jets and jet constituents (subjets) lists of [[pT],[eta],[phi]]. We generate this files by running Pythia with SlowJets over an LHE file generated in Madgraph 5. 
# 2) We center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0).
# 3) We shift the coordinates of each jet constituent so that the jet is centered at the origin in the new coordinates.
# 4) We calculate DeltaR for the subjets in the shifted coordinates and the angle theta for the principal axis.
# 5) We rotate the coordinate system so that the principal axis is the same direction (+ eta) for all jets.
# 6) We scale the pixel intensities such that sum_{i,j} I_{i,j}=1
# 7) We create the array of pT for the jet constituents, where each entry represents a pixel. We add all the jet constituents that fall within the same pixel.
# 8) We subtract the mean mu_{i,j} from each image, transforming each pixel intensity as I_{i,j}=I_{i,j}-mu_{i,j}.
# 9) We standardize the images adding a factor "bias" for noise suppression: Divide each pixel by the standard deviation of that pixel value among all the images in the training data set 
# 10) We reflect the image to ensure the 3rd maximum is on the right half-plane
# 11) We output a tuple with the numpy arrays and true value of the images that we will use as input for our neural network
# 12) We plot all the images.
# 13) We add the images to get the average jet image for all the events.
# 14) We plot the averaged image.
# May 15, 2017. Sebastian Macaluso
# Last updated: June 10, 2017.
# Written for Python 3.6.0


#To run this script:
# python image_preprocess2.py jets_subjets_directory
#Example: python image_preprocess2.py tt_5000_500_500
# python image_preprocess_bg-sig_std.py results_tt_10000_slowjet_200.0_1400.0_700_1.0 results_QCD_dijet_10000_780_800_700_1.0
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

np.set_printoptions(threshold=np.nan)

import scipy 

import time
start_time = time.time()

import image_preprocess_functions_color as fns

##---------------------------------------------------------------------------------------------
# Global variables

# local_dir='/Users/sebastian/Documents/Deep-Learning/jet_images/'
local_dir=''

print("Reading command file...")

commandfile=sys.argv[1]

with open(commandfile) as f:
   commands=f.readlines()

commands = [x.strip().split('#')[0].split() for x in commands] 

batch_label=''
myMethod=''
std_label='none'
bias=0.00000000000001
bias_label=''
myN_jets=1000000000000000000000000000000000000000
preprocessfile=''
stdfilename=''
stdmtx=np.array([0])
preprocess_label=''
mergeflag=1 # 1=use merged jets only; 0=include non-merged jets
merge_label='merged'

for command in commands:
  if len(command)>=2:
    if(command[0]=='LABEL'):
       dir_label=command[1]
    elif(command[0]=='BATCH'):
       batch_label=command[1]
    elif(command[0]=='NIMAGES'):
       myN_jets=int(command[1])
    elif(command[0]=='USEFILE'):
       preprocessfile=command[1]
    elif(command[0]=='USESTDFILE'):
       stdfilename=command[1]
    elif(command[0]=='NPOINTS'):
       npoints=int(command[1])
    elif(command[0]=='DRETA'):
       DReta=float(command[1])
    elif(command[0]=='DRPHI'):
       DRphi=float(command[1])
    elif(command[0]=='PREPROCESS'):
       preprocess_label=command[1]
    elif(command[0]=='MERGE'):
       mergeflag=int(command[1])


if(mergeflag==0):
    merge_label='unmerged'

print(commands)

dir_jets_subjets='pyroot/Output_'+dir_label+'/' #+batch_label
# dir_jets_subjets='../pyroot/Output_'+dir_label+'/' #+batch_label
std_dir='standardization/'

os.system('mkdir -p image_array')
os.system('mkdir -p plots')
Images_dir=local_dir+'plots/' #Output dir to save the image plots
image_array_dir=local_dir+'image_array/' #Output dir to save the image arrays



N_pixels=np.power(npoints-1,2)

ncolors=5

# Load std file if requested
if(stdfilename!=''):
  stdmtx=np.load(std_dir+stdfilename)
  print('Using standardization matrix from file',stdfilename)
#  print(stdmtx)
  if('sig' in stdfilename):
    std_label='sig'
  elif('bg' in stdfilename):
    std_label='bg'
  elif('all' in stdfilename):
    std_label='all'
  else:
    print('Error! std label not recognized!')
    
  if('std' in stdfilename):
    std_label=std_label+'_std'
  elif('n_moment' in stdfilename):
    std_label=std_label+'_n_moment'
  elif('max' in stdfilename):
    std_label=std_label+'_max'


##---------------------------------------------------------------------------------------------
#FUNCTIONS
##---------------------------------------------------------------------------------------------

  
##---------------------------------------------------------------------------------------------
##---------------------------------------------------------------------------------------------
#/////////////////////////////   MAIN FUNCTIONS    //////////////////////////////////
##---------------------------------------------------------------------------------------------



##---------------------------------------------------------------------------------------------
#9) Standardize the images with a bias for noise suppression: Divide each pixel by the standard deviation of that pixel value among all the images in the training data set 
##--------------------------------
################## USE STANDARD DEVIATION FROM ANOTHER SET OF IMAGES  ##################
##--------------------------------
def standardize_bias_std_other_set(Image, input_std_bias): 
  
#  print('-----------'*10)
#  print('-----------'*10)
#  print('Standardizing image with std from another set and a noise suppression factor ...')
#  print('-----------'*10)
# ADD AS LAST COLUMN THE UNSTANDARDIZED GRAYSCALE IMAGE TO HOPEFULLY PRESERVE MASS INFORMATION
#  std_im_list=[[pixel[0],(np.concatenate((pixel[1]/input_std_bias[pixel[0][0],pixel[0][1]],[pixel[1][0]+pixel[1][1]]))).tolist() ]  for pixel in Image]
  std_im_list=[[pixel[0],(pixel[1]/input_std_bias[pixel[0][0],pixel[0][1]]).tolist() ]  for pixel in Image]

#  print('Standardized images with shape (npoints-1 x npoints-1)(1st 2 image arrays)=\n {}'.format(std_im_list[0:2]))
#  print('-----------'*10)
#  print('-----------'*10)
  return std_im_list

def dxyresfunc(pt):
# input can be numpy array

# resolution in mm from fig 15 of 1405.6569, digitized and fit to a simple function 
  resmicrons=(11.4202+68.6757/np.power(pt,0.981246))*0.001
  
  return resmicrons
  
  
def jet_mass(Tracks,Towers):

  p0=np.sum(Tracks[0]*np.cosh(Tracks[1]))+np.sum(Towers[0]*np.cosh(Towers[1]))
  p1=np.sum(Tracks[0]*np.cos(Tracks[2]))+np.sum(Towers[0]*np.cos(Towers[2]))
  p2=np.sum(Tracks[0]*np.sin(Tracks[2]))+np.sum(Towers[0]*np.sin(Towers[2]))
  p3=np.sum(Tracks[0]*np.sinh(Tracks[1]))+np.sum(Towers[0]*np.sinh(Towers[1]))

  minv=np.max([p0*p0-p1*p1-p2*p2-p3*p3,0])
  minv=np.sqrt(minv)/175. # normalize it to the top mass
  
  return minv

  
# string should be either tt or qcd
def getnjets(dir_subjets,string):
  print('Calculating number of jets to preprocess',string)

  testout=subprocess.check_output('grep ", 1]" '+dir_subjets+'/subjets*'+string+'*.dat| wc -l',shell=True)
  testout=testout.strip().split(' ')
  Ntotjets=int(testout[0])
  
  if(mergeflag==0):
    testout=subprocess.check_output('grep ", 0]" '+dir_subjets+'/subjets*'+string+'*.dat| wc -l',shell=True)
    testout=testout.strip().split(' ')
    Ntotjets=Ntotjets+int(testout[0])
    
  return Ntotjets

# string should be either qcd or tt
def preprocess(dir_subjets,string,preprocess_label,std_mtx,N_jets,truthval,outputfile):

  print('Loading files for subjets')
  print('Subjet array format ([[[pTsubj1],[pTsubj2],...],[[etasubj1],[etasubj2],...],[[phisubj1],[phisubj2],...]])')
  print('-----------'*10)
  

  subjetlist = [filename for filename in np.sort(os.listdir(dir_subjets)) if ('subjets' in filename and string in filename and filename.endswith('.dat'))]
  N_analysis=len(subjetlist)
  print('Number of subjet files =',N_analysis)
  print('Loading subjet files...  \n {}'.format(subjetlist))

  images=[]
  jetmasslist=[]

  Ntotjets=0


    
  for ifile in range(N_analysis):
#    print(myN_jets,Ntotjets)

    if(Ntotjets>N_jets):
       break
       
    for s in open(dir_subjets+'/'+subjetlist[ifile]):
       rawsubjets=json.loads(s)
       jetmergeflag=rawsubjets[1]
       rawsubjets=rawsubjets[0]
       
       if(jetmergeflag>=mergeflag):
         Ntotjets=Ntotjets+1
         if(Ntotjets>N_jets):
             break
       
         if Ntotjets%10000==0:
           print('Already generated jet images for {} jets'.format(Ntotjets))
           elapsed=time.time()-start_time
           print('elapsed time',elapsed)    
              
         towerlist = [vec for vec in rawsubjets if len(vec)==3]
         tracklist = [vec for vec in rawsubjets if len(vec)>3]

#         print(len(tracklist),len(towerlist))         

         towerpTarray=np.array([])
         toweretaarray=np.array([])
         towerphiarray=np.array([])
         if(len(towerlist)>0):
           towerlist2=np.transpose(np.array(towerlist))
           towerpTarray=towerlist2[0]
           toweretaarray=towerlist2[1]
           towerphiarray=towerlist2[2]

         trackpTarray=np.array([]) 
         tracketaarray=np.array([]) 
         trackphiarray=np.array([]) 
         trackmuonarray=np.array([])
#         trackiparray=np.array([]) 
         trackchargearray=np.array([])         
         if(len(tracklist)>0):
           tracklist2=np.transpose(np.array(tracklist))
           
           trackpTarray=tracklist2[0]
           tracketaarray=tracklist2[1]
           trackphiarray=tracklist2[2]
           trackmuonarray=tracklist2[3]
           trackchargearray=tracklist2[4]
#           trackxarray=tracklist2[4]
#           trackyarray=tracklist2[5]

#           trackpxarray=trackpTarray*np.cos(trackphiarray);
#           trackpyarray=trackpTarray*np.sin(trackphiarray);

#           trackdxarray=trackpyarray*(trackpyarray*trackxarray-trackpxarray*trackyarray)/(trackpTarray*trackpTarray)
#           trackdyarray=trackpxarray*(trackpxarray*trackyarray-trackpyarray*trackxarray)/(trackpTarray*trackpTarray)

#           resarray=dxyresfunc(trackpTarray)
#           randomsmear=resarray*np.random.randn(len(trackdxarray))
#           trackiparray=np.abs(np.sqrt(trackdxarray*trackdxarray+trackdyarray*trackdyarray)+randomsmear)
#           trackiparray=trackiparray*0.5*(np.sign(2.-trackiparray)+1.)*0.5*(np.sign(trackpTarray-1.)+1.)
#           trackiparray=trackiparray/resarray
       
#         if(ifile<10):
#           print(Ntotjets)
#           print(trackiparray)
       


         trackarray=[trackpTarray,tracketaarray,trackphiarray,trackmuonarray,trackchargearray]
         towerarray=[towerpTarray,toweretaarray,towerphiarray]
       
         jetmass=jet_mass(trackarray,towerarray)
        
         preprocessed_image=fns.preprocess_color_image(towerarray,trackarray,DReta,DRphi,npoints,preprocess_label)

         if(std_mtx.any()):
           std_img=standardize_bias_std_other_set(preprocessed_image,std_mtx)
         else: 
           std_img=preprocessed_image


         print([std_img,truthval,jetmass],file=outputfile)
 #      print([images_bg[ijet],1,jetmasslist_bg[ijet]],file=bgfile)


#       images.append(std_img)
#       jetmasslist.append(jetmass)



  print('-----------'*10)
  print('Finished preprocessing images')
  print('Nimages = {}'.format(Ntotjets)) 
  print('-----------'*10)
 
#  return
   
#  return images,jetmasslist,preprocess_label



  
    

if __name__=='__main__':

    nsig=getnjets(dir_jets_subjets,'tt')
    nbg=getnjets(dir_jets_subjets,'qcd')
    myN_jets=np.min([nsig,nbg,myN_jets])
    print('Nsig, Nbg, and Njets to preprocess:',nsig,nbg,myN_jets)
#    myN_jets=10

#    preprocess_label='rot_vflip_hflip'
    outputfilename=str(myN_jets)+'_'+str(npoints-1)+'_'+dir_label+'_'+preprocess_label
    sigfile=open(image_array_dir+'tt_'+outputfilename+'_'+std_label+'_ncolors'+str(ncolors)+'_'+batch_label+'_'+merge_label+'.dat','w')
    bgfile=open(image_array_dir+'QCD_'+outputfilename+'_'+std_label+'_ncolors'+str(ncolors)+'_'+batch_label+'_'+merge_label+'.dat','w')
    

    print('Preprocessing and standardizing images...')
    preprocess(dir_jets_subjets,'tt',preprocess_label,stdmtx,myN_jets,0,sigfile)
    preprocess(dir_jets_subjets,'qcd',preprocess_label,stdmtx,myN_jets,1,bgfile)
    elapsed=time.time()-start_time
    print('elapsed time',elapsed)
 

#    print('OUTPUT PREPROCESSED AND STANDARDIZED IMAGES...')


#    output(images_sig,'tt_'+outputfilename+std_label)
#    output(images_bg,'QCD_'+outputfilename+std_label)
   
    
#    print('begin printing ALL images')
#    for ijet in range(myN_jets):
#        if ijet%10000==0:
#           print('Written out {} jet images'.format(ijet))
#    print('end printing ALL images')
    


# TRY CENTERING THE IMAGES AGAIN
#    print('re-centering the images after standardization')
#    sig_image_norm=center_images(sig_image_norm)
#    bg_image_norm=center_images(bg_image_norm)
#    elapsed=time.time()-start_time
#    print('elapsed time',elapsed)

# OUTPUT
  


    print('FINISHED.')

    print('-----------'*10)
    print("Code execution time = %s minutes" % ((time.time() - start_time)/60))
    print('-----------'*10) 
  
  
##-----------------------------------------------------------   
##-----------------------------------------------------------    
# Images specifications:
# ------------------------
# 300<pTj<400
# 100<jetMass<250
# eta_max=2.5
# treshold=0.95
# DeltaR=1.6
# Pixels=37 (size per pixel ~ HCal resolution in DeltaR ~ 0.0875)  
# ------------------------ 

  
  
  
