#!/usr/bin/env python

import numpy as np
import time


##---------------------------------------------------------------------------------------------
#2) We want to center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0). So we calculate eta_center,phi_center
def center(tracks,towers):
# format for subjets: [[pT1,pT2,...],[eta1,eta2,....],[phi1,phi2,....]]

#  print('Calculating the image center for the total pT weighted centroid pixel is at (eta,phi)=(0,0) ...')
#  print('-----------'*10)

#  Nsubjets=len(Subjets[0])
#  print len(tracks),len(towers)
#  print len(tracks[0]),len(towers[0])
 
  pTj=np.sum(tracks[0])+np.sum(towers[0])

  eta_c=(np.sum(tracks[0]*tracks[1])+np.sum(towers[0]*towers[1]))/pTj 
  phi_c=(np.sum(tracks[0]*tracks[2])+np.sum(towers[0]*towers[2]))/pTj 

  return eta_c,phi_c




##---------------------------------------------------------------------------------------------
#3) We shift the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates
def shift(subjets,Eta_c,Phi_c):
#  print('Shifting the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates ...')
#  print('-----------'*10)

  subjets[1]=subjets[1]-Eta_c
  subjets[2]=subjets[2]-Phi_c

  return subjets
  
  

##---------------------------------------------------------------------------------------------
#4) We calculate DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis
def principal_axis(tracks,towers):
#  print('Getting DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis ...')
#  print('-----------'*10)

  tan_theta=0.
  M11=np.sum(tracks[0]*tracks[1]*tracks[2])+np.sum(towers[0]*towers[1]*towers[2])
  M20=np.sum(tracks[0]*tracks[1]*tracks[1])+np.sum(towers[0]*towers[1]*towers[1])
  M02=np.sum(tracks[0]*tracks[2]*tracks[2])+np.sum(towers[0]*towers[2]*towers[2])  
  denom=(M20-M02+np.sqrt(4*M11*M11+(M20-M02)*(M20-M02)))
  if(denom!=0):
    tan_theta=2*M11/denom


  return tan_theta


##---------------------------------------------------------------------------------------------
#5b) We rotate the coordinate system so that the principal axis is the same direction (+ eta) for all jets
def rotate(subjets,tan_theta):
#  print('Rotating the coordinate system so that the principal axis is the same direction (+ eta) for all jets ...')
#  print('-----------'*10)

  rotpt=subjets[0]
  roteta=subjets[1]*np.cos(np.arctan(tan_theta))+subjets[2]*np.sin(np.arctan(tan_theta))
  rotphi=np.unwrap(-subjets[1]*np.sin(np.arctan(tan_theta))+subjets[2]*np.cos(np.arctan(tan_theta)))

  return [rotpt,roteta,rotphi,subjets[3],subjets[4]]


##---------------------------------------------------------------------------------------------
#6) We scale the pixel intensities such that sum_{i,j} I_{i,j}=1
def normalize(tracks,towers):
#  print('Scaling the pixel intensities such that sum_{i,j} I_{i,j}=1 ...')
#  print('-----------'*10)

  pTj=np.sum(tracks[0])+np.sum(towers[0])

  tracks[0]=tracks[0]/pTj
  towers[0]=towers[0]/pTj

  return tracks,towers



##---------------------------------------------------------------------------------------------
#7) We create a coarse grid for the array of pT for the jet constituents, where each entry represents a pixel. We add all the jet constituents that fall within the same pixel 
def create_color_image(tracks,towers,DReta,DRphi,npoints):
    
  ncolors=5

  etamin, etamax = -DReta, DReta # Eta range for the image
  phimin, phimax = -DRphi, DRphi # Phi range for the image

  allimages=[]
  grid=np.zeros((npoints-1,npoints-1,ncolors))
  nonzerogrid=np.zeros((npoints-1,npoints-1))
    
  ietalisttrack=((tracks[1]+DReta)/(2*DReta/float(npoints-1))).astype(int)
  iphilisttrack=((tracks[2]+DRphi)/(2*DRphi/float(npoints-1))).astype(int)
  ietalisttower=((towers[1]+DReta)/(2*DReta/float(npoints-1))).astype(int)
  iphilisttower=((towers[2]+DRphi)/(2*DRphi/float(npoints-1))).astype(int)

  for ipos in range(len(tracks[0])):
#     norm=1/float(len(tracks[0]))
     norm=1
     if(0<=ietalisttrack[ipos]<npoints-1 and 0<=iphilisttrack[ipos]<npoints-1):
#       if(tracks[4][ipos]<=2):
#          ipadd=[0,0,0,0,0,0]
#       elif(2<tracks[4][ipos]<=4):
#          ipadd=[norm,0,0,0,0,0]
#       elif(4<tracks[4][ipos]<=6):
#          ipadd=[0,norm,0,0,0,0]
#       elif(6<tracks[4][ipos]<=8):
#          ipadd=[0,0,norm,0,0,0]
#       elif(8<tracks[4][ipos]<=10):
#          ipadd=[0,0,0,norm,0,0]
#       elif(10<tracks[4][ipos]<=12):
#          ipadd=[0,0,0,0,norm,0]
#       else:
#          ipadd=[0,0,0,0,0,norm]
       toadd=[tracks[0][ipos],0,1,tracks[3][ipos],tracks[4][ipos]]
#       toadd.extend(ipadd)
       grid[ietalisttrack[ipos],iphilisttrack[ipos]]=grid[ietalisttrack[ipos],iphilisttrack[ipos]]+toadd

#       grid[ietalisttrack[ipos],iphilisttrack[ipos]][4]=np.max([grid[ietalisttrack[ipos],iphilisttrack[ipos]][4],])
       nonzerogrid[ietalisttrack[ipos],iphilisttrack[ipos]]=1.
  for ipos in range(len(towers[0])):
     if(0<=ietalisttower[ipos]<npoints-1 and 0<=iphilisttower[ipos]<npoints-1):
       toadd=[0,towers[0][ipos],0,0,0]
#       toadd.extend([0,0,0,0,0,0])
       grid[ietalisttower[ipos],iphilisttower[ipos]]=grid[ietalisttower[ipos],iphilisttower[ipos]]+toadd
       nonzerogrid[ietalisttower[ipos],iphilisttower[ipos]]=1.
	 
  test_output=np.nonzero(nonzerogrid)
  test_output2=grid[test_output[0],test_output[1]].tolist()
  test_output3=np.transpose([test_output[0],test_output[1]]).tolist()
  test_output4=[list(a) for a in zip(test_output3,test_output2)]
#  print(test_output4)
#We ask some treshold for the total pT fraction to keep the image when some constituents fall outside of the range for (eta,phi)
  sum=np.sum(grid)
  if sum<0.95:
    print('Error! image intensity below threshold!',sum)
#    print(ietalisttrack)
#    print(ietalisttower)
#    print(tracks[2])
#    print(iphilisttrack)
#    print(towers[2])
#    print(iphilisttower)


  final_image=test_output4

  return final_image



##---------------------------------------------------------------------------------------------
#10)Reflect the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane
def ver_flip_color(Image,npoints): 
  
#  print('Flipping the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane ...')
#  print('-----------'*10)
  half_img=(npoints-2)/2

  left_img=[pixel[1][0:2] for pixel in Image if pixel[0][0]<half_img ]
  right_img=[pixel[1][0:2] for pixel in Image if pixel[0][0]>half_img ]
     
  left_sum=np.sum(left_img)
  right_sum=np.sum(right_img)

  if left_sum<right_sum:
      flip_image = [[[npoints-2-pixel[0][0],pixel[0][1]],pixel[1]] for pixel in Image]
  else:
      flip_image = Image

#  print('-----------'*10)
#  print('-----------'*10)
  return flip_image  


##---------------------------------------------------------------------------------------------
#10b)Reflect the image with respect to the horizontal axis to ensure the 3rd maximum is on the upper half-plane
def hor_flip_color(Image,npoints): 
  
#  print('Flipping the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane ...')
#  print('-----------'*10)

  half_img=(npoints-2)/2

  lower_img=[pixel[1][0:2] for pixel in Image if pixel[0][1]<half_img]
  upper_img=[pixel[1][0:2] for pixel in Image if pixel[0][1]>half_img]
     
  lower_sum=np.sum(lower_img)
  upper_sum=np.sum(upper_img)

  if lower_sum>upper_sum:
    flip_image = [[[pixel[0][0],npoints-2-pixel[0][1]],pixel[1]] for pixel in Image]
  else:
    flip_image = Image

  return flip_image  
  
def preprocess_color_image(towerarray,trackarray,DReta,DRphi,npoints,preprocess_label):
 
  preprocess_cmnd=preprocess_label.split('_')

  trackpTarray=trackarray[0]
  tracketaarray=trackarray[1]
  trackphiarray=trackarray[2]
  trackmuonarray=trackarray[3]
  trackchargearray=trackarray[4]
   
  towerpTarray=towerarray[0]
  toweretaarray=towerarray[1]
  towerphiarray=towerarray[2]
  towermuonarray=np.zeros(len(towerarray[0]))
  towerchargearray=np.zeros(len(towerarray[0]))
  
  if(len(trackphiarray)>0):
    refphi=trackphiarray[0]
  else:
    refphi=towerphiarray[0]
          
# make sure the phi values are each the correct branch 
#  print(refphi)
#  print(trackphiarray)
  trackphiarray1=np.dstack((trackphiarray-refphi,trackphiarray-refphi-2*np.pi,trackphiarray-refphi+2*np.pi,trackphiarray-refphi-4*np.pi,trackphiarray-refphi+4*np.pi))[0]
  indphi=np.transpose(np.argsort(np.abs(trackphiarray1),axis=1))[0]
  trackphiarray2=trackphiarray1[np.arange(len(trackphiarray1)),indphi]+refphi
#  print(trackphiarray2)

#  print(towerphiarray)  
  towerphiarray1=np.dstack((towerphiarray-refphi,towerphiarray-refphi-2*np.pi,towerphiarray-refphi+2*np.pi,towerphiarray-refphi-4*np.pi,towerphiarray-refphi+4*np.pi))[0]
  indphi=np.transpose(np.argsort(np.abs(towerphiarray1),axis=1))[0]
  towerphiarray2=towerphiarray1[np.arange(len(towerphiarray1)),indphi]+refphi
#  print(towerphiarray2)

  TrackSubjets=[trackpTarray,tracketaarray,trackphiarray2,trackmuonarray,trackchargearray]
  TowerSubjets=[towerpTarray,toweretaarray,towerphiarray2,towermuonarray,towerchargearray]
#       AllSubjets=[np.concatenate((TrackSubjets[0],TowerSubjets[0])),np.concatenate((TrackSubjets[1],TowerSubjets[1])),np.concatenate((TrackSubjets[2],TowerSubjets[2]))]

# begin preprocessing


  eta_c, phi_c=center(TrackSubjets,TowerSubjets)  
  TrackSubjets=shift(TrackSubjets,eta_c,phi_c)
  TowerSubjets=shift(TowerSubjets,eta_c,phi_c)

  if('rot' in preprocess_cmnd):
    tan_theta=principal_axis(TrackSubjets,TowerSubjets) 
    TrackSubjets=rotate(TrackSubjets,tan_theta)
    TowerSubjets=rotate(TowerSubjets,tan_theta)

  if('norm' in preprocess_cmnd):
    TrackSubjets,TowerSubjets=normalize(TrackSubjets,TowerSubjets)

  raw_image=create_color_image(TrackSubjets,TowerSubjets,DReta,DRphi,npoints)  

  if('vflip' in preprocess_cmnd):
    raw_image=ver_flip_color(raw_image,npoints)  

  if('hflip' in preprocess_cmnd):
    raw_image=hor_flip_color(raw_image,npoints)  
 
  return raw_image
 
def output(images,outputfilename):
# type label is typically either tt or qcd
# preprocess label indicates which preprocessing steps were followed
# std label indicates what standardization was performed (or none)

  print("Saving data in .npy format ...")

  np.save(outputfilename+'.npy',images)
  print('List of jet image arrays filename = {}'.format(outputfilename+'.npy'))
  print('-----------'*10)

#  output_image_array_data_true_value(images,dir_label+'_'+batch_label+'_'+myMethod+bias_label+'_'+std_label+'_'+preprocess_label)   
#  elapsed=time.time()-start_time
#  print('elapsed time',elapsed)

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# NOT USED BELOW THIS LINE
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

##---------------------------------------------------------------------------------------------
#2) We want to center the image so that the total pT weighted centroid pixel is at (eta,phi)=(0,0). So we calculate eta_center,phi_center
def center_images(images):
# format for subjets: [[pT1,pT2,...],[eta1,eta2,....],[phi1,phi2,....]]

#  print('Calculating the image center for the total pT weighted centroid pixel is at (eta,phi)=(0,0) ...')
#  print('-----------'*10)

#  Nsubjets=len(Subjets[0])
  centeredimages=[]
  for image in images:
     pTarray=[np.sum(pixel[1][0:2]) for pixel in image]
     rgbarray=[pixel[1] for pixel in image]
     etaarray=np.array([-cf.DR+2*cf.DR/(2*float(npoints-1))+pixel[0][0]*2*cf.DR/float(npoints-1) for pixel in image])
     phiarray=np.array([-cf.DR+2*cf.DR/(2*float(npoints-1))+pixel[0][1]*2*cf.DR/float(npoints-1) for pixel in image])
     eta_c=np.sum(etaarray*pTarray)/np.sum(pTarray)     
     phi_c=np.sum(phiarray*pTarray)/np.sum(pTarray)     
#     print(eta_c,phi_c)
     etaarray=etaarray-eta_c
     phiarray=phiarray-phi_c
     ieta=((etaarray+cf.DR)/(2*cf.DR/float(npoints-1))).astype(int)
     iphi=((phiarray+cf.DR)/(2*cf.DR/float(npoints-1))).astype(int)

     ietaphi=[list(a) for a in zip(ieta,iphi)]
#     print(ietaphi)
     centeredimage=[list(a) for a in zip(ietaphi,rgbarray)]
#     print(centeredimage)
     centeredimages.append(centeredimage)
     
  return centeredimages

##---------------------------------------------------------------------------------------------
#8) We subtract the mean mu_{i,j} of each image, transforming each pixel intensity as I_{i,j}=I_{i,j}-mu_{i,j}
def zero_center(Image,ref_image):
  print('Subtracting the mean mu_{i,j} of each image, transforming each pixel intensity as I_{i,j}=I_{i,j}-mu_{i,j} ...')
  print('-----------'*10)
  mu=[]
  Im_sum=[]
  N_pixels=np.power(npoints-1,2)
#  for ijet in range(0,len(Image)):
#    mu.append(np.sum(Image[ijet])/N_pixels)
#     Im_sum.append(np.sum(Image[ijet]))
#   print('Mean values of images= {}'.format(mu))
#   print('Sum of image pT (This should ideally be 1 as the images are normalized except when some jet constituents fall outside of the image range )= {}'.format(Im_sum))
  zeroImage=[]
  for ijet in range(0,len(Image)):# As some jet images were discarded because the total momentum of the constituents within the range of the image was below the treshold, we use len(image) instead of Njets
    if ijet==10:
        for i in range(37):
           for j in range(37):
              print("zero_center image")
              print(i,j,Image[ijet][i,j])
              print("ref image")
              print(i,j,ref_image[i,j])
              print("diff")
              print((Image[ijet]-ref_image)[i,j])


    zeroImage.append(Image[ijet]-ref_image)
#    print(ijet,mu[ijet])

  print('Grid after subtracting the mean (1st 2 images)= \n {}'.format(Image[0:2])) 
  print('-----------'*10)
#   print('Mean of first images',mu[0:6])
  return zeroImage 

  
##---------------------------------------------------------------------------------------------
#12) We plot all the images
def plot_all_images(Image, type):
  
#   for ijet in range(0,len(Image)):
  for ijet in range(1200,1230):
    imgplot = plt.imshow(Image[ijet], 'gnuplot', extent=[-cf.DR, cf.DR,-cf.DR, cf.DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
    plt.xlabel('$\eta^{\prime\prime}$')
    plt.ylabel('$\phi^{\prime\prime}$')
  #plt.show()
    fig = plt.gcf()
    plt.savefig(Images_dir+'1jet_images/Im_'+str(name)+'_'+str(npoints-1)+'_'+str(ijet)+'_'+type+'.png')
#   print(len(Image))
#   print(type(Image[0]))



##---------------------------------------------------------------------------------------------
#13) We add the images to get the average jet image for all the events
def add_images(Image):
  print('Adding the images to get the average jet image for all the events ...')
  print('-----------'*10)
  N_images=len(Image)
#   print('Number of images= {}'.format(N_images))
#   print('-----------'*10)
  avg_im=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image
  for ijet in range(0,len(Image)):
    avg_im=avg_im+Image[ijet]
    #avg_im2=np.sum(Image[ijet])
  print('Average image = \n {}'.format(avg_im))
  print('-----------'*10)
#  print('Average image 2 = \n {}'.format(avg_im2))
  #We normalize the image
  Total_int=np.absolute(np.sum(avg_im))
  print('Total intensity of average image = \n {}'.format(Total_int))
  print('-----------'*10)
#  norm_im=avg_im/Total_int
  norm_im=avg_im/N_images
#   print('Normalized average image (by number of images) = \n {}'.format(norm_im))
  print('Normalized average image = \n {}'.format(norm_im))
  print('-----------'*10)
  norm_int=np.sum(norm_im)
  print('Total intensity of average image after normalizing (should be 1) = \n {}'.format(norm_int))
  return norm_im
  
  
##---------------------------------------------------------------------------------------------
#13b) We add the images to get the average jet image for all the events
def add_stacked_images(Image):
  print('Adding the images to get the average jet image for all the events ...')
  print('-----------'*10)
  N_images=len(Image)
#   print('Number of images= {}'.format(N_images))
#   print('-----------'*10)
  avg_im=np.zeros(((npoints-1)*2,npoints-1)) #create an array of zeros for the image
  # avg_im=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image
  for ijet in range(0,len(Image)):
    avg_im=avg_im+Image[ijet]
    #avg_im2=np.sum(Image[ijet])
  print('Average image = \n {}'.format(avg_im))
  print('-----------'*10)
#  print('Average image 2 = \n {}'.format(avg_im2))
  #We normalize the image
  Total_int=np.absolute(np.sum(avg_im))
  print('Total intensity of average image = \n {}'.format(Total_int))
  print('-----------'*10)
#   norm_im=avg_im/Total_int
# Each image in the stacked structure was normalized to 1, so we divide by the number of images to normalize the stacked image to 2 (1 from each subimage)
  norm_im=avg_im/N_images
#   print('Normalized average image (by number of images) = \n {}'.format(norm_im))
  print('Normalized average image = \n {}'.format(norm_im))
  print('-----------'*10)
  norm_int=np.sum(norm_im)
#   print('Total intensity of average image after normalizing (should be 1) = \n {}'.format(norm_int))
  return norm_im, N_images
  
  
##---------------------------------------------------------------------------------------------
#14) We plot the averaged image
def plot_avg_image(Image, type,name,Nimages):
  print('Plotting the averaged image ...')
  print('-----------'*10)
#   imgplot = plt.imshow(Image[0], 'viridis')# , origin='upper', interpolation='none', vmin=0, vmax=0.5)  
  imgplot = plt.imshow(Image, 'gnuplot', extent=[-cf.DR, cf.DR,-cf.DR, cf.DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
  plt.xlabel('$\eta^{\prime\prime}$')
  plt.ylabel('$\phi^{\prime\prime}$')
  fig = plt.gcf()
  image_name=str(name)+'_avg_im_'+str(Nimages)+'_'+str(npoints-1)+'_'+type+'_'+sample_name+'.png'
  plt.savefig(Images_dir+image_name)
  print('Average image filename = {}'.format(Images_dir+image_name))

##---------------------------------------------------------------------------------------------
#14) We plot the stacked averaged image 
def plot_stacked_avg_image(Image, type,name,Nimages):
  print('Plotting the averaged image ...')
  print('-----------'*10)
#   imgplot = plt.imshow(Image[0], 'viridis')# , origin='upper', interpolation='none', vmin=0, vmax=0.5)  
  imgplot = plt.imshow(Image, 'gnuplot', extent=[-cf.DR, cf.DR,-2*cf.DR, 2*cf.DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
#   imgplot = plt.imshow(Image[0])
#   plt.show()
  plt.xlabel('$\eta^{\prime\prime}$')
  plt.ylabel('$\phi^{\prime\prime}$')
  fig = plt.gcf()
  image_name=str(name)+'_avg_im_'+str(Nimages)+'_'+str(npoints-1)+'_'+type+'_'+sample_name+'.png'
  plt.savefig(Images_dir+image_name)
  print('Average image filename = {}'.format(Images_dir+image_name))
  

