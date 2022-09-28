# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 8:35:27 2021

@author: liuyang20
"""

import numpy as np
import scipy.ndimage as spim
from porespy.tools import extend_slice
from porespy import settings
from porespy.tools import get_tqdm
tqdm = get_tqdm()

#%%


import scipy.ndimage as ndimage
struct = ndimage.generate_binary_structure(3, 2)

def SegmentPoresandThroats(net,im,dt,voxel_size):
    
    '''
    This function is to identy the attribution of a voxel to which pore or throat
    Returns the hydraulic conduit lengths, elextrical conduit lengths, and volume
    
    Parameters
    ----------
    net : dict
        A dictionary in OpenPNM containing all the pore and throat size data, which is obtained by snow5. snow5 is a modified version of snow2 in the Porespy
        
    im : ndarray,
        the pore regions segmented by porespy, from 1 to Np
        
    dt : distance map
        the distance value corresponding to each voxel. the same shape as im
            
    voxel_size : scalar (default = 1)
        The resolution of the image

    Returns
    -------
    net : dict
        A dictionary in OpenPNM which contains hydraulic conduit lengths(net['throat.Lt_hydraulic']),
        elextrical conduit lengths (net['throat.Lt_electrical']),
        and volumes of pores and throats (net['pore.volume_seg'],net['throat.volume_seg'])
        
    '''
    
    slices = spim.find_objects(im)  # Get 'slices' into im for each pore region
    p_coords_dt = net['pore.voxelcoords_dtglobal'].copy()
    t_coords = net['throat.voxelcoords_center'].copy()
    p_coords_dt = np.array(p_coords_dt,dtype=int)
    t_coords = np.array(t_coords,dtype=int)
    
    Rts = net['throat.inscribed_diameter']/voxel_size/2
    Rps = net['pore.extended_diameter']/voxel_size/2
    
    Np = len(p_coords_dt)
    Nt = len(net['throat.conns'])
    
    im_volume = im.copy()  #the image used to record voxels' attribution and for calculation of volume
    
    
    for x in range(im.ndim):
        mask = p_coords_dt[:,x]>=im.shape[x]
        p_coords_dt[mask,x] = im.shape[x]-1

        mask = t_coords[:,x]>=im.shape[x]
        t_coords[mask,x] = im.shape[x]-1
        
    net['throat.Lt_hydraulic'] = np.zeros((Nt,3),dtype=float)  #data set of hydraulic conduit lengths for pore1, pore2 and throat respectively.
    net['throat.Lt_electrical'] = np.zeros((Nt,3),dtype=float)  #data set of electrical conduit lengths for pore1, pore2 and throat respectively.
    
    msg = "Pore-Throat Segmentation"
    
    Ts = np.arange(Nt)
    ts = tqdm(Ts, desc=msg, **settings.tqdm)
    
    for t in ts:
    
        p1 = net['throat.conns'][t,0]
        p2 = net['throat.conns'][t,1]
        
        coord_p1_abs = (p_coords_dt[p1, :]).reshape(3,1)
        # coord_p1_abs = coord_p1_abs.astype(int)
        coord_t_abs = t_coords[t]
        coord_t_abs = np.array(coord_t_abs,dtype=int).reshape(3,1)
        coord_p2_abs = (p_coords_dt[p2, :]).reshape(3,1)
        # coord_p2_abs = coord_p2_abs.astype(int)
        
        
        if (coord_p2_abs==coord_t_abs).all() or (coord_p1_abs==coord_t_abs).all():
            
            # the center of pore1 or pore2 overlaps with the center of throat
            Lt = 0.0
            L_subp1 = get_leng_betw2ps(coord_p1_abs, coord_t_abs)
            L_subp2 = get_leng_betw2ps(coord_p2_abs, coord_t_abs)
            net['throat.Lt_hydraulic'][t] = np.array([L_subp1,Lt,L_subp2])
            net['throat.Lt_electrical'][t] = np.array([L_subp1,Lt,L_subp2])
            continue
            
        if np.linalg.norm(coord_p1_abs-coord_p2_abs)==0:
            # the center of pore1 verlaps with the center of pore2 
            coord_p1_abs = (net['pore.p_coords_dt_local'][p1, :]).reshape(3,1)
            coord_p2_abs = (net['pore.p_coords_dt_local'][p2, :]).reshape(3,1)
        
        
        #reg12 is the minimal BOX that includes the regions at the ends of throat 't'
        #regs12, the voxels in pore1 in labeled as 1, the voxels in pore2 in labeled as 2, 
        #dt_regs12, distance values corresponding to voxels in regs12
        regs12,dt_regs12,inds_reg12 = get_reg12(num_reg1=p1+1,num_reg2=p2+1,
                                                    regions=im,dt_regions=dt,
                                                    slice_regs=slices)
        
        #determine the coordinate of pore1,pore2 and throat in the BOX before affine transformations
        coord_p1_1,coord_p2_1,coord_t_1 = get_center_coord_1(inds_reg12,coord_p1_abs,
                                                                 coord_p2_abs,coord_t_abs)
        
        #num_t and num_reg are numbers to distinguish teh voxels belonging to a pore or throat
        # num_t and num_reg can take on any two different values, 6 and 1 in this case
        num_t = 6
        num_reg = 1
        
        #reg1 is has the same shape as regs12, whereas only the region1 is labeled as 1 and others are 0
        reg1 = get_sing_reg(regs12,num_reg)
        reg1[coord_t_1[0,0],coord_t_1[1,0],coord_t_1[2,0]] = num_t # the voxel at the throat center in reg1 is labeled as num_t
        
        mask1 = regs12==num_reg
        reg1_dia = spim.binary_dilation(input=mask1, structure=struct)
        mask_interface = regs12*reg1_dia
        interface = np.where(mask_interface==2) # interface is the set of voxels that saparating two regions. 
        reg1[interface] = num_t  # the voxels at the interface are labeled as num_t, which are the initial section of dilation process
        
        #the coordinates of voxels corresponding to pore1  in BOX before affine transformations 
        coords_reg1_1 = get_regs_mask(reg1) # 3 rows correspond to the X,Y,Z coordinates in the BOX, respectively.
        mask_coord_p1 = (coords_reg1_1 == coord_p1_1).all(axis=0) # the mask where the pore1 center is marked sa True while others are False
        mask_coord_t = (coords_reg1_1 == coord_t_1).all(axis=0) # the mask where the throat center is marked sa True while others are False
        
        reg1_numlabel =  np.full_like(reg1,-1,dtype=int)
        X_reg1_1 = coords_reg1_1[0]
        Y_reg1_1 = coords_reg1_1[1]
        Z_reg1_1 = coords_reg1_1[2]
        elems_reg1 = np.arange(len(mask_coord_p1))
        reg1_numlabel[X_reg1_1,Y_reg1_1,Z_reg1_1] = elems_reg1
        
        #the affine  matrix that makes the centers of pore1 and pore2 along the vertical axis
        R1 =  Rotate_matrix(coord_t_1,coord_p1_1)
        #To rotate the BOX around the throat center by affine transformations.
        #After the operation, the vector from the throat center to the pore1 center in the new BOX is along VERTICAL direction (0,0,1)
        #coords_reg1_2 is the coords of voxels in region1 after affine transformations.
        coords_reg1_2 = rotate_im(R1,coords_reg1_1,coord_t_1)
        
        coord_p1_2 = coords_reg1_2[:,mask_coord_p1]
        coord_t_2 = coords_reg1_2[:,mask_coord_t]
        
        #the hight of centers of pore1 and throat along vertical axis after affine transformations.
        zp1_2 = coord_p1_2[2,0]
        zt_2 = coord_t_2[2,0]
        
        if zp1_2<zt_2:
            
            #The image should be rotated 180 degrees again to make p1 over t (zp1_2>zt_2)
            coords_reg1_2 = -coords_reg1_2
            coord_p1_2 = coords_reg1_2[:,mask_coord_p1]
            coord_t_2 = coords_reg1_2[:,mask_coord_t]
            
            zp1_2 = coord_p1_2[2,0]
            zt_2 = coord_t_2[2,0]
        
        #The to get the distance value corresponding to void voxels in reg1
        dt_reg1_elems = get_dt_sub_reg(coords_reg1_1,dt_regs12)
        
        R_t = Rts[t]
        R_p1 = Rps[p1]
            
        Zs_reg1_2 = coords_reg1_2[2] # the vertical height corresponding to voxels in the new BOX after rotation
        
        #the direct length between centers of pore1 and throat
        Lp1t = get_leng_betw2ps(coord_t_1,coord_p1_1)
        
        #To determine the attribution of voxels in region1 to pore1 or throat
        #Le_subt1 and Le_subp1 are electrical conduit lengths of the "half throat" and  "pore1", respectively
        #L_subt1 and L_subp1 are hydraulic conduit lengths of the "half throat" and  "pore1", respectively
        #t_elems_reg1 indicate the voxels belong to the "half throat"
        Le_subt1,Le_subp1,L_subt1,L_subp1,t_elems_reg1 = GetConduitLengths(Lp1t,reg1,num_t,num_reg,R_p1,R_t,
                                                                                 dt_reg1_elems,reg1_numlabel,
                                                                                 zp1_2,coords_reg1_1,Zs_reg1_2,coord_p1_1)
        
        X_start = inds_reg12[0][0].start
        Y_start = inds_reg12[0][1].start
        Z_start = inds_reg12[0][2].start
        s_offset = np.array([X_start,Y_start,Z_start]).reshape(3,1)  #the coordinates of the base of the BOX  in the full image 
        
        coords_ts_reg1_1 = coords_reg1_1[:,t_elems_reg1]
        coords_ts_abs_1 = coords_ts_reg1_1+s_offset
        X_ts = coords_ts_abs_1[0,:]
        Y_ts = coords_ts_abs_1[1,:]
        Z_ts = coords_ts_abs_1[2,:]
        
        im_volume[X_ts,Y_ts,Z_ts] = Np+t+1  ###in the im_volume, the voxels belong to t is relabeled as Np+t+1
        
        
        ####### the process for the "throat-pore2" is exactly the same as "throat-pore1"
        num_reg=2
        reg2 = get_sing_reg(regs12,num_reg)
        reg2[coord_t_1[0,0],coord_t_1[1,0],coord_t_1[2,0]] = num_t
        reg2[interface] = num_t
        
        coords_reg2_1 = get_regs_mask(reg2)
        mask_coord_p2 = (coords_reg2_1 == coord_p2_1).all(axis=0)
        mask_coord_t = (coords_reg2_1 == coord_t_1).all(axis=0)
        
        reg2_numlabel =  np.full_like(reg2,-1,dtype=int)
        X_reg2_1 = coords_reg2_1[0]
        Y_reg2_1 = coords_reg2_1[1]
        Z_reg2_1 = coords_reg2_1[2]
        elems_reg2 = np.arange(len(mask_coord_p2))
        reg2_numlabel[X_reg2_1,Y_reg2_1,Z_reg2_1] = elems_reg2
        
        R2 =  Rotate_matrix(coord_t_1,coord_p2_1)
        coords_reg2_2 = rotate_im(R2,coords_reg2_1,coord_t_1)
        
        coord_p2_2 = coords_reg2_2[:,mask_coord_p2]
        coord_t_2 = coords_reg2_2[:,mask_coord_t]
        
        zp2_2 = coord_p2_2[2,0]
        zt_2 = coord_t_2[2,0]
        
        if zp2_2<zt_2:
            coords_reg2_2 = -coords_reg2_2
            coord_p2_2 = coords_reg2_2[:,mask_coord_p2]
            coord_t_2 = coords_reg2_2[:,mask_coord_t]
            
            zp2_2 = coord_p2_2[2,0]
            zt_2 = coord_t_2[2,0]
        
        dt_reg2_elems = get_dt_sub_reg(coords_reg2_1,dt_regs12)
        
        R_t = Rts[t]
        R_p2 = Rps[p2]
            
        
        Zs_reg2_2 = coords_reg2_2[2]
        Lp2t = get_leng_betw2ps(coord_t_1,coord_p2_1)
        
        Le_subt2,Le_subp2,L_subt2,L_subp2,t_elems_reg2 = GetConduitLengths(Lp2t,reg2,num_t,num_reg,R_p2,R_t,
                                                                dt_reg2_elems,reg2_numlabel,
                                                                zp2_2,coords_reg2_1,Zs_reg2_2,coord_p2_1)
        
        coords_ts_reg2_1 = coords_reg2_1[:,t_elems_reg2]
        coords_ts_abs_1 = coords_ts_reg2_1+s_offset
        X_ts = coords_ts_abs_1[0,:]
        Y_ts = coords_ts_abs_1[1,:]
        Z_ts = coords_ts_abs_1[2,:]
        
        im_volume[X_ts,Y_ts,Z_ts] = Np+t+1  ###in the im_volume, the voxels belong to t is relabeled as Np+t+1
        
        
        Lt_hydraulic = L_subt1+L_subt2
        Lt_electrical = Le_subt1+Le_subt2
        
        #the throat conduit length is assumed to be larger than 1 voxel size
        Lt_hydraulic = max(Lt_hydraulic,1)
        Lt_electrical = max(Lt_electrical,1)
        
        
        
        net['throat.Lt_electrical'][t] = [Le_subp1,Le_subp2,Lt_electrical]
        net['throat.Lt_hydraulic'][t] = [L_subp1,L_subp2,Lt_hydraulic]
        
    net['throat.Lt_electrical'] = net['throat.Lt_electrical']*voxel_size
    net['throat.Lt_hydraulic'] = net['throat.Lt_hydraulic']*voxel_size
    
    #Calculation of volume of throats and pores
    Vs_nondim = np.zeros(Np+Nt,dtype=float)
    slices = spim.find_objects(im_volume)
    for i in np.arange(Np+Nt):
        if slices[i] is None:
            continue
        
        s = extend_slice(slices[i], im_volume.shape)
        sub_im = im_volume[s].copy()
        unit_im = (sub_im==(i+1))
        Vs_nondim[i] = np.sum(unit_im)
    
    net['pore.volume_seg'] = (Vs_nondim[:Np])*voxel_size**3
    net['throat.volume_seg'] = (Vs_nondim[Np:(Np+Nt)])*voxel_size**3
    
    return net
    




#%%

def Rotate_matrix(coord_p1_1,coord_p2_1):
    
    '''
    To calculate the affine matrix
    '''
    
    x = (coord_p2_1-coord_p1_1).reshape(3)
    norm_x = np.linalg.norm(x)
    
    if norm_x==0:
        print('Error pore1 overlap pore2, cannot get solution')
        return
    
    if x[0]==0 and x[1]==0:
        R = np.eye(3)
        return R
    y = norm_x*np.array([0,0,1])
    norm_y = np.linalg.norm(y)
    
    if norm_x != norm_y:
        return
    
    cos_theta = np.dot(x,y)/norm_x/norm_y
    sin_theta = np.linalg.norm(np.cross(x,y))/norm_x/norm_y
    
    k = np.cross(x,y)
    k = k/np.linalg.norm(k)
    k = k.reshape(3,1)
    
    K = np.zeros((3,3))
    K[0,1] = -k[2]
    K[1,0] = k[2]
    K[0,2] = k[1]
    K[2,0] = -k[1]
    K[1,2] = -k[0]
    K[2,1] = k[0]
    
    
    R = np.eye(3)*cos_theta+(1-cos_theta)*np.kron(k,k.T)+sin_theta*K
    
    return R


def get_reg12(num_reg1,num_reg2,regions,dt_regions,slice_regs):
    '''
    This function finds the smallest hexahedral subregion of the original image that contains region1 and region2
    
    
    '''
    
    inds_reg12 = get_slicesofreg12(slice_regs,num_reg1,num_reg2)
    regs12 = (regions[inds_reg12[0]]).copy()   
    
    mask_reg1 = regs12==num_reg1
    mask_reg2 = regs12==num_reg2
    not_mask_reg12 = ~(mask_reg1|mask_reg2)
    
    regs12[not_mask_reg12] = 0
    regs12[mask_reg1] = 1
    regs12[mask_reg2] = 2
    
    dt_regs12 = (dt_regions[inds_reg12[0]]).copy()
    dt_regs12[not_mask_reg12] = 0.0
    
    
    return [regs12,dt_regs12,inds_reg12]
    

def get_center_coord_1(slices_regs12,coord_p1_abs,coord_p2_abs,coord_t_abs):
    '''
    This function finds the center coords of pore1, pore2 and throat before affine transformation
    '''
    
    X_start = slices_regs12[0][0].start
    Y_start = slices_regs12[0][1].start
    Z_start = slices_regs12[0][2].start

    s_offset = np.array([X_start,Y_start,Z_start]).reshape(3,1)
    
    coord_p1_1 = coord_p1_abs-s_offset
    coord_p2_1 = coord_p2_abs-s_offset
    coord_t_1 = coord_t_abs-s_offset
    
    return [coord_p1_1,coord_p2_1,coord_t_1]

def get_regs_mask(reg12):
    '''
    The function is to obtain the coordinates of the voxel  corresponding to the void
    '''
    
    coords_1 = np.argwhere(reg12>0).astype(int)
    coords_1 = coords_1.T

    return coords_1
    
def rotate_im(R,coords_1,coord_p1_1):
    '''
    The function is to rotate the sub-image to make the centers of pore and throat along the vertical direction
    '''
    if coords_1.shape[0]!=3 or coord_p1_1.shape[0]!=3:
        return
    
    coords_rela2p1_1 = (coords_1-coord_p1_1).astype(float)
    coords_rela2p1_2 = np.dot(R,coords_rela2p1_1)
    
    coords_rela2p1_2 = np.round(coords_rela2p1_2,decimals=4)
    
    Xmin = np.min(coords_rela2p1_2[0])
    Ymin = np.min(coords_rela2p1_2[1])
    Zmin = np.min(coords_rela2p1_2[2])
    
    coords_2 = coords_rela2p1_2-np.array([Xmin,Ymin,Zmin]).reshape(3,1)
    
    return coords_2

def get_center_coord_2(coords_1,coords_2,coord_p1_1,coord_p2_1,coord_t_1):
    '''
    This function finds the center coords of pore1, pore2 and throat after affine transformation
    '''
    
    mask_coord_p1 = (coords_1 == coord_p1_1).all(axis=0)
    mask_coord_p2 = (coords_1 == coord_p2_1).all(axis=0)
    mask_coord_t = (coords_1 == coord_t_1).all(axis=0)

    coord_p1_2 = coords_2[:,mask_coord_p1]
    coord_p2_2 = coords_2[:,mask_coord_p2]
    coord_t_2 = coords_2[:,mask_coord_t]
        
    Z_p1_2 = coord_p1_2[2,0]
    Z_p2_2 = coord_p2_2[2,0]
    Z_t_2 = coord_t_2[2,0]
    
    return [[Z_p1_2,Z_p2_2,Z_t_2],
            [mask_coord_p1,mask_coord_p2,mask_coord_t]]


def get_dt_sub_reg(coords_1,dt_regs_1):
    '''
    The function is to obtain the distance value of void voxels in the sub-image
    '''
    
    X_1 = coords_1[0]
    Y_1 = coords_1[1]
    Z_1 = coords_1[2]
    
    dt_elems = dt_regs_1[X_1,Y_1,Z_1]
    
    
    return dt_elems


from numpy import sqrt

def get_leng_betw2ps(coord_p1,coord_p2):
    
    '''
    The function is to calculate the direct length between pores
    '''
    
    arrow = coord_p1-coord_p2
    arrow = arrow**2
    leng = sqrt(np.sum(arrow))
    
    return leng

def get_slicesofreg12(Slices_regions,reg1_num,reg2_num):
    '''
    The function to find the indexice of the minimum hexahedral matrix that contains region1,2 from the overall image
    '''
    
    s_reg1 = Slices_regions[reg1_num-1]
    s_reg2 = Slices_regions[reg2_num-1]
    
    
    X_start_reg12 = min(s_reg1[0].start,s_reg2[0].start)
    X_end_reg12 = max(s_reg1[0].stop,s_reg2[0].stop)
    Y_start_reg12 = min(s_reg1[1].start,s_reg2[1].start)
    Y_end_reg12 = max(s_reg1[1].stop,s_reg2[1].stop)
    Z_start_reg12 = min(s_reg1[2].start,s_reg2[2].start)
    Z_end_reg12 = max(s_reg1[2].stop,s_reg2[2].stop)
    
    s_reg12 = [(slice(X_start_reg12,X_end_reg12,None),
               slice(Y_start_reg12,Y_end_reg12,None),
               slice(Z_start_reg12,Z_end_reg12,None))]
    
    return s_reg12


def get_sing_reg(regs12,target_reg):
    '''
    The function is to make the value of voxels of the target region becom 1 in the sub-image , while ithers become 0
    '''
    mask = regs12==target_reg
    regs_targ = regs12.copy()
    regs_targ[~mask] = 0
    
    return regs_targ
   
def throat_dia(reg,num_t,num_reg):
    '''
    the function is to get a new section during one dilation of throat voxles
    
    '''
    
    struct = ndimage.generate_binary_structure(3, 2)
    mask_dia = spim.binary_dilation(reg==num_t,structure=struct)
    mask_t = mask_dia*(reg==num_reg)
    
    reg[mask_t] = num_t
    
    return reg

def get_r_dia(t_elems,dt_elems):
    '''
    the function is to get the hydraulic radius of the new section after dilation
    '''
    dt_throat = dt_elems[t_elems]
    
    R = max(dt_throat)
    
    return R

def get_z_dia(t_elems,Zs_reg_2):
    '''
    the function is to get the vertical height of the new section
    '''
    Z_max = max(Zs_reg_2[t_elems])
    
    elem_Zmax = t_elems[np.argmax(Zs_reg_2[t_elems])]
    return [Z_max,elem_Zmax]

def get_telems_dia(reg,reg_numlabel,num_t):
    '''
    the function is to get the newly involved throat voxels
    '''
    mask_t = reg==num_t
    t_elems = reg_numlabel[mask_t]
    
    return t_elems

def GetConduitLengths(Lpt,reg,num_t,num_reg,Rp,Rt,
                        dt_reg_elems,reg_numlabel,Zp,
                        coords_reg_1,Zs_reg_2,coord_p_1):
    '''
    This function is to identify the attribution of voxels to pore or throat
    
    Parameters
    ----------
    Lpt : float
        the direct length from the throat center to the pore center.
    reg : numpy.ndarray int
        the smallest BOX contains regions of pore1 and pore2.
        if the "throat-pore1" is processed, voxels corresponds to pore region1 and the interface are labeled as 1 and num_t, respectively
    num_t : int
        the label of voxels corresponding to the interface between pore region1 and pore region2 in reg.
    num_reg : float
        the label of voxels corresponding to the pore region in reg.
    Rp : float
        the radius of the pore.
    Rt : float
        the radius of the throat.
    dt_reg_elems : float
        the distance value of voxels in reg.
    reg_numlabel : numpy.ndarray int
        the same shape with reg.
        the voxels belong to the pore region1 and the interface are marked by a serial number form 0, while the other voxels are labeled as -1.
    coords_reg_1 : numpy.ndarray int
        the coordinates of voxels corresponding to pore1 and the interface in BOX before affine transformations 
        3 rows correspond to the X,Y,Z coordinates, respectively.
    Zs_reg_2 : float
        the vertical height (along Z direction) of voxels in the new BOX with affine transformation.
    coord_p_1 : float
        the coordinate of center of pore1.
    
    Returns
    -------
    Le_t : float
        electrical conduit length of the "half throat"
    Le_p : float
        electrical conduit length of the pore
    L_t : float
        hydraulic conduit length of the "half throat"
    L_p : float
        hydraulic conduit length of the pore
        
    t_elems : numpy.array
    the voxels belong to the "half throat"
        
    '''
    
    R_excess = Rp-Rt
    mask = reg==num_t
    p_label = reg_numlabel[coord_p_1[0],coord_p_1[1],coord_p_1[2]]
    
    ###The pore raidus may be smaller than throat radius in SNOW, though scarcely
    if R_excess<0:
        Le_t = 0.5*Lpt
        Le_p = Lpt-Le_t
        Lt = 0.5*Lpt 
        Lp = Lpt-Lt
        mask = reg==num_t
        t_elems = reg_numlabel[mask]
        
        return [Le_t,Le_p,Lt,Lp,t_elems]
    
    ###Normally, the pore raidus is larger than throat radius
    ###Then the voxels attribution is determined by local resistance
    reg_dia_i = reg.copy()
    elem_Rmax = []
    t_elems_newAdd = get_telems_dia(reg_dia_i,reg_numlabel,num_t)
    Rmax_newtelems,Newthroat_Rmax = get_Rmax_NewtElem(t_elems_newAdd,dt_reg_elems)
    elem_Rmax.append(Newthroat_Rmax)
    
    Rsist_new = 1/Rmax_newtelems**4
    Elec_Rsist_new = 0.0
    
    A_newtelems = len(t_elems_newAdd)
    if A_newtelems>0:
        Elec_Rsist_new += 1/A_newtelems
    At = A_newtelems
    Ap_max = At
    
    dia_num = 0
    mask_dia_num = -np.ones_like(reg,dtype=int)
    coords_t_elems_newAdd = coords_reg_1[:,t_elems_newAdd]
    mask_dia_num[coords_t_elems_newAdd[0],coords_t_elems_newAdd[1],coords_t_elems_newAdd[2]] = dia_num
    dia_num += 1
    
    flag = True
    
    R_p_max = Rmax_newtelems
    R_p_max =max(R_p_max,Rp)
    R_t_min = Rmax_newtelems
    R_t_min = min(R_t_min,Rt)
    
    R_t_along_max = Rt
    
    #the maximal height of sections
    Zmax = max(Zs_reg_2[t_elems_newAdd])
    
    while flag:
        
        t_elems_beforeDia = get_telems_dia(reg_dia_i,reg_numlabel,num_t)
        reg_dia_i = throat_dia(reg,num_t,num_reg)
        t_elems_afterDia = get_telems_dia(reg_dia_i,reg_numlabel,num_t)
        t_elems_newAdd = np.setdiff1d(t_elems_afterDia,t_elems_beforeDia)
        
        #if no voxels are involved, then stop
        if len(t_elems_newAdd)==0:
            flag = False
            break
        
        Rmax_newtelems,Newthroat_Rmax = get_Rmax_NewtElem(t_elems_newAdd,dt_reg_elems)
        
        R_p_max = max(R_p_max,Rmax_newtelems)
        R_t_min = min(R_t_min,Rmax_newtelems)
        
        R_t_along_max = max(R_t_along_max,Rmax_newtelems)
        
        elem_Rmax.append(Newthroat_Rmax)
        
        Rsist_new = Rsist_new+1/R_t_along_max**4
        Zmax = max(Zmax,max(Zs_reg_2[t_elems_newAdd]))
        
        A_newtelems = len(t_elems_newAdd)
        if A_newtelems>0:
            Elec_Rsist_new += 1/A_newtelems
            Ap_max = max(Ap_max,A_newtelems)
        
        coords_t_elems_newAdd = coords_reg_1[:,t_elems_newAdd]
        mask_dia_num[coords_t_elems_newAdd[0],coords_t_elems_newAdd[1],coords_t_elems_newAdd[2]] = dia_num
        dia_num += 1
        
        #if the maximal height of sections is higher than the pore center
        #or the pore center is involved, then stop
        if Zmax>=Zp or p_label in t_elems_newAdd:
            flag = False
            break

    elem_Rmax = np.array(elem_Rmax,dtype=int)
    
    #Electrical length
    W2_t = 1/At
    W2_p = 1/Ap_max
    if W2_t==W2_p:
        W2_p = 0.99*W2_t
        
    res_tem = W2_p*dia_num
    if res_tem>=Elec_Rsist_new:
        res_tem=0.99*Elec_Rsist_new
    dia_range1 = (Elec_Rsist_new-res_tem)/(W2_t-W2_p)
    if dia_range1>=dia_num:
        dia_range1 = dia_num-1
        
    Le_t = dia_range1/dia_num*Lpt   
    Le_p = Lpt-Le_t
    
    #Hydraulic length
    W1_t = 1/Rt**4
    W1_p = 1/R_p_max**4
    
    if W1_t==W1_p:
        W1_p = 0.99*W1_t
    
    res_tem = W1_p*dia_num
    if res_tem>=Rsist_new:
        res_tem = 0.99*Rsist_new
        
    dia_range = (Rsist_new-res_tem)/(W1_t-W1_p)
    if dia_range>=dia_num:
        dia_range = dia_num-1
    
    Lt = dia_range/dia_num*Lpt
    Lp = Lpt-Lt
    
    mask_t=(mask_dia_num<=dia_range)*(mask_dia_num>=0)
    t_elems = reg_numlabel[mask_t]
    
    
    
    return [Le_t,Le_p,Lt,Lp,t_elems]


def projLength(coord_vertex,coord_base,coord_lim):
    '''
    the function is to calculate the projection length between pore centers to a specified direction
    '''
    
    coord_vertex = coord_vertex.reshape(3)
    coord_base = coord_base.reshape(3)
    coord_lim = coord_lim.reshape(3)
    
    L_pt = get_leng_betw2ps(coord_vertex,coord_base)
    
    if L_pt!=0:
        
        x = coord_lim-coord_vertex
        y = coord_base-coord_vertex
        L_plim = np.dot(x,y)/L_pt
        if L_plim<=0:
            L_plim = get_leng_betw2ps(coord_vertex,coord_lim)
            
    else:
        L_plim = get_leng_betw2ps(coord_vertex,coord_lim)
    
    return abs(L_plim)
    

def get_Rmax_NewtElem(Newt_elems,dt_elems):
    '''
    the function is to obtain the hydraulic radius of a new section
    '''
    
    dt_Newthroat = dt_elems[Newt_elems]
    Rmax_newtelems = max(dt_Newthroat)
    Newthroat_Rmax = Newt_elems[np.argmax(dt_Newthroat)]
    
    return [Rmax_newtelems,Newthroat_Rmax]



#%%

from porespy.networks import add_boundary_regions
from porespy.networks import label_phases, label_boundaries
from porespy.filters import snow_partitioning, snow_partitioning_parallel
from porespy.tools import Results
from loguru import logger


def snow5(phases,
          phase_alias=None,
          boundary_width=3,
          accuracy='standard',
          voxel_size=1,
          sigma=0.4,
          r_max=4,
          parallelization={},):
    r"""
    Applies the SNOW algorithm to each phase indicated in ``phases``.

    This function is a combination of ``snow`` [1]_, ``snow_dual`` [2]_,
    ``snow_n`` [3]_, and ``snow_parallel`` [4]_ from previous versions.

    Parameters
    ----------
    phases : ndarray
        An image indicating the phase(s) of interest. A watershed is
        produced for each integer value in ``phases`` (except 0's). These
        are then combined into a single image and one network is extracted
        using ``regions_to_network``.
    phase_alias : dict
        A mapping between integer values in ``phases`` and phase name
        used to add labels to the network. For instance, asssuming a
        two-phase image, ``{1: 'void', 2: 'solid'}`` will result in the
        labels ``'pore.void'`` and ``'pore.solid'``, as well as
        ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``. If not provided, aliases are assumed to be
        ``{1: 'phase1', 2: 'phase2, ...}``.  Phase labels can also be
        applied afterward using ``label_phases``.
    boundary_width : depends
        Number of voxels to add to the beginning and end of each axis.
        This argument can either be a scalar or a list. If a scalar is
        passed, it will be applied to the beginning and end of all axes.
        In case of a list, you can specify the number of voxels for each
        axis individually. Here are some examples:

            - [0, 3, 0]: 3 voxels only applied to the y-axis.

            - [0, [0, 3], 0]: 3 voxels only applied to the end of y-axis.

            - [0, [3, 0], 0]: 3 voxels only applied to the beginning of y-axis.

        The default is to add 3 voxels on both ends of all axes. For each
        boundary width that is not 0, a label will automatically be
        applied indicating which end of which axis (i.e. ``'xmin'`` and
        ``'xmax'``).
    accuracy : string
        Controls how accurately certain properties are calculated during
        the analysis of regions in the ``regions_to_network`` function.
        Options are:

            - 'standard' (default)
                Computes the surface areas and perimeters by simply
                counting voxels. This is *much* faster but does not
                properly account for the rough voxelated nature
                of the surfaces.

            - 'high'
                Computes surface areas using the marching cube
                method, and perimeters using the fast marching method. These
                are substantially slower but better account for the
                voxelated nature of the images.

    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side
        of a voxel, so the volume of a voxel would be **voxel_size**-cubed.
    r_max : int
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : float
        The standard deviation of the Gaussian filter used in step 1. The
        default is 0.4.  If 0 is given then the filter is not applied.
    parallelization : dict
        The arguments for controlling the parallelization of the watershed
        function are rolled into this dictionary, otherwise the function
        signature would become too complex. Refer to the docstring of
        ``snow_partitioning_parallel`` for complete details. If no values
        are provided then the defaults for that function are used here.
        To disable parallelization pass ``parallel=None``, which will
        invoke the standard ``snow_partitioning`` or ``snow_partitioning_n``.

    Returns
    -------
    network : Results object
        A custom object is returned with the following data added as attributes:

        - 'phases'
            The original ``phases`` image with any padding applied

        - 'regions'
            The watershed segmentation of the image, including boundary
            regions if padding was applied

        - 'network'
            A dictionary containing all the extracted network properties in
            OpenPNM format ('pore.coords', 'throat.conns', etc).

    References
    ----------
    .. [1] Gostick JT. Versatile and efficient pore network extraction
       method using marker-based watershed segmentation. Phys. Rev. E. 96,
       023307 (2017)
    .. [2] Khan ZA, Tranter TG, Agnaou M, Elkamel A, and Gostick JT, Dual
       network extraction algorithm to investigate multiple transport
       processes in porous materials: Image-based modeling of pore and
       grain-scale processes. Computers and Chemical Engineering. 123(6),
       64-77 (2019)
    .. [3] Khan ZA, García-Salaberri PA, Heenan T, Jervis R, Shearing P,
       Brett D, Elkamel A, Gostick JT, Probing the structure-performance
       relationship of lithium-ion battery cathodes using pore-networks
       extracted from three-phase tomograms. Journal of the
       Electrochemical Society. 167(4), 040528 (2020)
    .. [4] Khan ZA, Elkamel A, Gostick JT, Efficient extraction of pore
       networks from massive tomograms via geometric domain decomposition.
       Advances in Water Resources. 145(Nov), 103734 (2020)

    """
    
    # print('phases shape is: ',phases.shape)
    
    regions = None
    for i in range(phases.max()):
        phase = phases == (i + 1)
        if parallelization is not None:
            
            snow = snow_partitioning_parallel(
                im=phase, sigma=sigma, r_max=r_max, **parallelization)
        else:
            
            snow = snow_partitioning(im=phase, sigma=sigma, r_max=r_max)   #snow是执行完dt产生regions后的结果
        if regions is None:
            regions = np.zeros_like(snow.regions, dtype=int)
            
        # Note: Using snow.regions > 0 here instead of phase is needed to
        # handle a bug in snow_partitioning, see issue #169 and #430
        regions += snow.regions + regions.max()*(snow.regions > 0)
    
    if phases.shape != regions.shape:
        logger.warning(f"Image was cropped to {regions.shape} during watershed")
        for ax in range(phases.ndim):
            phases = np.swapaxes(phases, 0, ax)
            phases = phases[:regions.shape[ax], ...]
            phases = np.swapaxes(phases, 0, ax)
    
    # Inspect and clean-up boundary_width argument
    boundary_width = _parse_pad_width(boundary_width, phases.shape)
    
    # If boundaries were specified, pad the images accordingly
    if np.any(boundary_width):
        regions = add_boundary_regions(regions, pad_width=boundary_width)
        phases = np.pad(phases, pad_width=boundary_width, mode='edge')
    
    # Perform actual extractcion on all regions
    net,im,dt = regions_to_network5(regions, phases=phases, accuracy=accuracy, voxel_size=voxel_size)
    # If image is multiphase, label pores/throats accordingly
    
    if phases.max() > 1:
        phase_alias = _parse_phase_alias(phase_alias, phases)
        net = label_phases(net, alias=phase_alias)
    # If boundaries were added, label them accordingly
    if np.any(boundary_width):
        W = boundary_width.flatten()
        L = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'][:phases.ndim*2]
        L = [L[i]*int(W[i] > 0) for i in range(len(L))]
        L = np.reshape(L, newshape=boundary_width.shape)
        net = label_boundaries(net, labels=L)
        
    result = Results()
    result.network = net
    result.phases = phases
    
    result.im = im
    result.dt = dt
    
    return result


def _parse_phase_alias(alias, phases):
    r"""
    """
    if alias is None:
        alias = {i+1: 'phase' + str(i+1) for i in range(phases.max())}
    for i in range(phases.max()):
        if i+1 not in alias.keys():
            alias[i+1] = 'phase'+str(i+1)
    return alias


def _parse_pad_width(pad_width, shape):
    r"""
    """
    ndim = len(shape)
    pad_width = np.atleast_1d(np.array(pad_width, dtype=object))

    if np.size(pad_width) == 1:
        pad_width = np.tile(pad_width.item(), ndim).astype(object)
    if len(pad_width) != ndim:
        raise Exception(f"pad_width must be scalar or {ndim}-element list")

    tmp = []
    for elem in pad_width:
        if np.size(elem) == 1:
            tmp.append(np.tile(np.array(elem).item(), 2))
        elif np.size(elem) == 2 and np.ndim(elem) == 1:
            tmp.append(elem)
        else:
            raise Exception("pad_width components can't have 2+ elements")

    return np.array(tmp)


from skimage.morphology import disk, ball
from edt import edt
from porespy.tools import get_tqdm, make_contiguous
from porespy.metrics import region_surface_areas, region_interface_areas
from porespy.metrics import region_volumes
tqdm = get_tqdm()

def regions_to_network5(regions, phases=None, voxel_size=1, accuracy='standard'):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    regions : ndarray
        An image of the material partitioned into individual regions.
        Zeros in this image are ignored.
    phases : ndarray, optional
        An image indicating to which phase each voxel belongs. The returned
        network contains a 'pore.phase' array with the corresponding value.
        If not given a value of 1 is assigned to every pore.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    accuracy : string
        Controls how accurately certain properties are calculated. Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels.  This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method.  These are substantially
            slower but better account for the voxelated nature of the images.

    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').

    Notes
    -----
    The meaning of each of the values returned in ``net`` are outlined below:
    
    'pore.region_label'
        The region labels corresponding to the watershed extraction. The
        pore indices and regions labels will be offset by 1, so pore 0
        will be region 1.
    'throat.conns'
        An *Nt-by-2* array indicating which pores are connected to each other
    'pore.region_label'
        Mapping of regions in the watershed segmentation to pores in the
        network
    'pore.local_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the pore region in isolation
    'pore.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.geometric_centroid'
        The center of mass of the pore region as calculated by
        ``skimage.measure.center_of_mass``
    'throat.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    
    'pore.region_volume'
        The volume of the pore region computed by summing the voxels
    
    'pore.volume'
        The volume of the pore found by as volume of a mesh obtained from the
        marching cubes algorithm
    'pore.surface_area'
        The surface area of the pore region as calculated by ?
    'throat.cross_sectional_area'
        The cross-sectional area of the throat found by ?
    'throat.perimeter'
        The perimeter of the throat found by ?
    'pore.inscribed_diameter'
        The diameter of the largest sphere inscribed in the pore region
    'pore.extended_diameter'
        ?
    'throat.inscribed_diameter'
        The diameter of the largest sphere inscribed the throat region
    'throat.total_length'
        ?
    'throat.direct_length'
        ?

    """
    logger.trace('Extracting pore/throat information')
    im = make_contiguous(regions)
    
    struc_elem = disk if im.ndim == 2 else ball
    voxel_size = float(voxel_size)
    
    if phases is None:
        phases = (im > 0).astype(int)
    if im.size != phases.size:
        raise Exception('regions and phase are different sizes, probably ' +
                        'because boundary regions were not added to phases')
    
    dt = edt(phases == 1)
    for i in range(2, phases.max()+1):
        dt += edt(phases == i)
    
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=int)
    p_area_surf = np.zeros((Np, ), dtype=int)
    p_phase = np.zeros((Np, ), dtype=int)
    # The number of throats is not known at the start, so lists are used
    # which can be dynamically resized more easily.
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []
    
    # Start extracting size information for pores and throats
    msg = "Extracting pore and throat properties"
    
    for i in tqdm(Ps, desc=msg, **settings.tqdm):
    
    
        pore = i - 1
        if slices[pore] is None:
            continue
        
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]  
        sub_dt = dt[s]  
        
        pore_im = sub_im == i   
        
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')  
        
        pore_dt = edt(padded_mask) 
        
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        
        p_coords_cm[pore, :] = spim.center_of_mass(pore_im) + s_offset
        
        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0] 
        p_coords_dt[pore, :] = temp + s_offset
        
        p_phase[pore] = (phases[s]*pore_im).max() 
        sub_dt_reg_i = sub_dt*pore_im
        temp = np.vstack(np.where(sub_dt_reg_i == sub_dt_reg_i.max()))[:, 0]
        p_coords_dt_global[pore, :] = temp + s_offset
        
        p_volume[pore] = np.sum(pore_im)
        p_dia_local[pore] = 2*np.amax(pore_dt)
        p_dia_global[pore] = 2*np.amax(sub_dt_reg_i)
        
        # The following is overwritten if accuracy is set to 'high'
        p_area_surf[pore] = np.sum(pore_dt == 1)
        
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        
        Pn = np.unique(im_w_throats)[1:] - 1
        
        for j in Pn:
            if j > pore:
                
                t_conns.append([pore, j])
                
                vx = np.where(im_w_throats == (j + 1))
                
                t_dia_inscribed.append(2*np.amax(sub_dt[vx])) 
                
                # The following is overwritten if accuracy is set to 'high'
                t_perimeter.append(np.sum(sub_dt[vx] < 2)) 
                
                # The following is overwritten if accuracy is set to 'high'
                t_area.append(np.size(vx[0]))
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                
                t_coords.append(tuple([t_inds[k][temp] for k in range(im.ndim)]))
    # Clean up values
    p_coords = p_coords_cm
    
    Nt = len(t_dia_inscribed)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords_cm.T, np.zeros((Np, )))).T
        t_coords = np.vstack((np.array(t_coords).T, np.zeros((Nt, )))).T

    net = {}
    ND = im.ndim
    
    net['throat.conns'] = np.array(t_conns)
    net['pore.coords'] = np.array(p_coords)*voxel_size
    net['pore.all'] = np.ones_like(net['pore.coords'][:, 0], dtype=bool)
    net['throat.all'] = np.ones_like(net['throat.conns'][:, 0], dtype=bool)
    
    net['pore.region_label'] = np.array(p_label)
    net['pore.phase'] = np.array(p_phase, dtype=int)
    net['throat.phases'] = net['pore.phase'][net['throat.conns']]
    
    V = np.copy(p_volume)*(voxel_size**ND)
    
    net['pore.region_volume'] = V  # This will be an area if image is 2D
    f = 3/4 if ND == 3 else 1.0
    
    net['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
    
    # Extract the geometric stuff
    net['pore.local_peak'] = np.copy(p_coords_dt)*voxel_size
    net['pore.global_peak'] = np.copy(p_coords_dt_global)*voxel_size
    net['pore.geometric_centroid'] = np.copy(p_coords_cm)*voxel_size
    net['throat.global_peak'] = np.array(t_coords)*voxel_size
    net['pore.inscribed_diameter'] = np.copy(p_dia_local)*voxel_size
    net['pore.extended_diameter'] = np.copy(p_dia_global)*voxel_size
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed)*voxel_size
    P12 = net['throat.conns']
    
    PT1 = np.sqrt(np.sum(((p_coords[P12[:, 0]]-t_coords)*voxel_size)**2,
                          axis=1))
    PT2 = np.sqrt(np.sum(((p_coords[P12[:, 1]]-t_coords)*voxel_size)**2,
                          axis=1))
    
    net['throat.total_length'] = PT1 + PT2
    
    dist = (p_coords[P12[:, 0]] - p_coords[P12[:, 1]])*voxel_size
    
    net['throat.direct_length'] = np.sqrt(np.sum(dist**2, axis=1))
    
    net['throat.perimeter'] = np.array(t_perimeter)*voxel_size
    
    if (accuracy == 'high') and (im.ndim == 2):
        logger.warning('High accuracy mode is not available in 2D, ' +
                       'reverting to standard accuracy')
        accuracy = 'standard'
    if (accuracy == 'high'):
        net['pore.volume'] = region_volumes(regions=im, mode='marching_cubes')
        areas = region_surface_areas(regions=im, voxel_size=voxel_size)
        net['pore.surface_area'] = areas
        interface_area = region_interface_areas(regions=im, areas=areas,
                                                voxel_size=voxel_size)
        A = interface_area.area
        net['throat.cross_sectional_area'] = A
        net['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)     
    else:
        net['pore.volume'] = np.copy(p_volume)*(voxel_size**ND)
        net['pore.surface_area'] = np.copy(p_area_surf)*(voxel_size**2)
        A = np.array(t_area)*(voxel_size**2)
        net['throat.cross_sectional_area'] = A
        net['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)
    
    net['pore.voxelcoords_dtglobal'] = np.array(p_coords_dt_global,dtype=int)
    net['pore.voxelcoords_dtlocal'] = np.array(p_coords_dt,dtype=int)
    net['pore.voxelcoords_geometric_centroid'] = np.around(net['pore.geometric_centroid']/voxel_size).astype(int)
    net['throat.voxelcoords_center'] = np.array(t_coords,dtype=int)
    
    return [net,im,dt]




