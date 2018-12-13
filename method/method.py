#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 14:36:00 2018

@author: sry
"""
import numpy as np

def block_lms_point_by_line(lms_frame,ang_frame,p1,p2,res):
    """
    p1:(2) [x,y]
    p2:(2) [x,y]
    lms_frame:(num_of_lms_point + ?,)
    res:(1,) resolution
    ang_frame:(num_of_lms_point,) increasing sequence, ang of each lms_point
    """
    if np.isinf(p1[0]) or np.isinf(p1[1]) or np.isinf(p2[0]) or np.isinf(p2[1]):
        return
    if np.isnan(p1[0]) or np.isnan(p1[1]) or np.isnan(p2[0]) or np.isnan(p2[1]):
        return


    theta1 = np.arctan2(p1[1],p1[0])
    theta2 = np.arctan2(p2[1],p2[0])
    H, = np.shape(ang_frame)
    
    #theta2 > theta1
    if theta1 > theta2:
        theta1,theta2 = theta2,theta1
        p1,p2 = p2,p1

        
    if theta2 - theta1 > np.pi:
        theta1,theta2 = theta2,theta1
        p1,p2 = p2,p1
        ind1 = (int)(np.ceil((theta1-ang_frame[0])/res) )
        ind2 = (int)(np.floor((theta2-ang_frame[0])/res) )
        ind = np.concatenate( [np.arange(ind1,H),np.arange(0,ind2+1)],axis = 0)
    else:
        ind1 = (int)(np.ceil((theta1-ang_frame[0])/res) )
        ind2 = (int)(np.floor((theta2-ang_frame[0])/res) )
        ind = np.arange(ind1,ind2+1)
 
        """
        -------------------------
        
               PI/2        []  ->
               
             ld\|/lu
         PI   -[*]-     0             [] ->
             rd/|\ru
             
              -PI/2 
        -------------------------
        """
        

    ang = ang_frame[ind]

    ang1 = ang - theta1
    ang2 = theta2 - ang

    l1 = np.sqrt(p1[0]*p1[0] + p1[1]*p1[1])
    l2 = np.sqrt(p2[0]*p2[0] + p2[1]*p2[1])
    lms_seg = l1*l2*np.sin(ang1+ang2)/(l1*np.sin(ang1)+l2*np.sin(ang2))
    lms_frame[ind] = np.min( [lms_seg,lms_frame[ind]] ,axis = 0)

"""
#demo usage of block_lms_point_by_line:
lms_frame = np.linspace(100.,100.,361)
ang_frame = np.linspace(-np.pi,np.pi,361)
p1 = np.array([10,5])
p2 = np.array([20,-5])
res = np.pi*2/360
block_lms_point_by_line(lms_frame,ang_frame,p1,p2,res)
p1 = np.array([5,-5])
p2 = np.array([15,-5])
block_lms_point_by_line(lms_frame,ang_frame,p1,p2,res)
"""



def creat_lms_frame(init_s,x_ego_frame,env_cars,num_of_lms_lines):
    """
    init_s['lw'] :(1,) left wall position
    init_s['rw'] :(1,) right wall position
    x_ego_frame:(3,...) x,y,yaw,???...
    env_cars:(num_of_cars,3)
    """
    
    """
    cars and lms definitions:
    suppose that shape of cars are squares
    suppose that the maximum range of lms is 100m
    """
    
    #default setting
    lms_frame = np.linspace(100.,100.,num_of_lms_lines+1) # '+1' in case of error
    ang_frame = np.linspace(-np.pi,np.pi,num_of_lms_lines)
    res = 2*np.pi / (num_of_lms_lines-1)
    theta = np.pi/4
    diagonal_len_half = np.power(2,0.5)
    """
    -------------------------

            y
        _________    
        | \     |
        |   \   |   ->  x
        |_theta_|
        
    -------------------------
    """    
    #calculate car corner
    x,y,yaw = x_ego_frame[:3]
    envx,envy,envyaw = np.split(env_cars,3,axis = 1)
    R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]) #by right mutiply this R, coordinate will shift to ego_cars
    #R_inv = np.linalg.inv( np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]) )
    relx,rely,relyaw = envx-x,envy-y,envyaw-yaw
    relx,rely = np.split( np.matmul(np.concatenate([relx,rely],axis = 1),R),2,axis = 1 )
    
    p_r_lu = np.concatenate([relx + diagonal_len_half*np.cos(relyaw+theta),rely + diagonal_len_half*np.sin(relyaw+theta)],axis = 1)
    p_r_ru = np.concatenate([relx + diagonal_len_half*np.cos(relyaw-theta),rely + diagonal_len_half*np.sin(relyaw-theta)],axis = 1)
    p_r_ld = np.concatenate([relx + diagonal_len_half*np.cos(relyaw + np.pi - theta),rely + diagonal_len_half*np.sin(relyaw + np.pi - theta)],axis = 1)
    p_r_rd = np.concatenate([relx + diagonal_len_half*np.cos(relyaw - np.pi + theta),rely + diagonal_len_half*np.sin(relyaw - np.pi + theta)],axis = 1)
    
    
    #block them by cars
    n,_ = np.shape(env_cars)
    for i in range(n):
        """
        -------------------------
        
               PI/2        []  ->
               
             ld\|/lu
         PI   -[*]-     0             [] ->
             rd/|\ru
             
              -PI/2 
        -------------------------
        """
        block_lms_point_by_line(lms_frame,ang_frame,p_r_lu[i],p_r_ru[i],res)
        block_lms_point_by_line(lms_frame,ang_frame,p_r_lu[i],p_r_ld[i],res)
        block_lms_point_by_line(lms_frame,ang_frame,p_r_rd[i],p_r_ld[i],res)
        block_lms_point_by_line(lms_frame,ang_frame,p_r_rd[i],p_r_ru[i],res)
        
    #block them by wall
    if init_s is not None:
        lw1 = np.array([-900,init_s['lw']]) - x_ego_frame[:2]
        lw2 = np.array([1100,init_s['lw']])- x_ego_frame[:2]
        rw1 = np.array([-900,init_s['rw']])- x_ego_frame[:2]
        rw2 = np.array([1100,init_s['rw']])- x_ego_frame[:2]
        block_lms_point_by_line(lms_frame,ang_frame,np.matmul(lw1,R),np.matmul(lw2,R),res)
        block_lms_point_by_line(lms_frame,ang_frame,np.matmul(rw1,R),np.matmul(rw2,R),res)
    
    return lms_frame

"""
#demo usage of creat_lms_frame:
x_ego_frame = (2,3,0)
env_cars = np.array([[5,6,0],[0,1,0]])
num_of_lms_lines = 361
init_s = {}
init_s['lw'] = 15.0 #distance to left wall
init_s['rw'] = -5.0
lms_frame = creat_lms_frame(init_s,x_ego_frame,env_cars,num_of_lms_lines)
"""

def x2lms(init_s,x_ego,env_cars,num_of_lms_lines = 361):
    T,_ = np.shape(x_ego)
    lms = np.zeros([T,num_of_lms_lines])
    env_cars = np.array(env_cars)
    for i in range(T):
        lms[i,:] = creat_lms_frame(init_s,x_ego[i,:3],env_cars[i,:,:],num_of_lms_lines)[:num_of_lms_lines]
    return lms
    
    

def v2x(v,yaw,dt,x0 = (0,0)):
    """
    v:1-d 
    yaw, 1-d
    dt:
    x0: (xx,xy) init. x
    """
    vy = np.sin(yaw)*v
    vx = np.cos(yaw)*v
    xy = np.cumsum(vy*dt)+x0[1]
    xx = np.cumsum(vx*dt)+x0[0]
    x = np.concatenate((np.expand_dims(xx,1),np.expand_dims(xy,1),np.expand_dims(yaw,1)),axis = 1)
    
    return x
    
def v2x_2(v,yaw,dt,x0 = (0,0)):
    """
    v:1-d 
    yaw, 1-d
    dt:
    x0: (xx,xy) init. x
    """
    vy = np.sin(yaw)*v
    vx = np.cos(yaw)*v
    xy = np.cumsum(vy*dt)+x0[1]
    xx = np.cumsum(vx*dt)+x0[0]
    x = np.concatenate((np.expand_dims(xx,1),np.expand_dims(xy,1),np.expand_dims(v,1),np.expand_dims(yaw,1)),axis = 1)
    
    return x
    
def x2yaw(x,y,dt):
    dy = np.diff(y,axis = -1)
    dx = np.diff(x,axis = -1)
    v = np.sqrt(dx**2+dy**2)/dt
    if np.size(np.shape(v)) == 1:
        v = np.concatenate((v,v[[-1]]),axis = 0 )
    else:
        v = np.concatenate((v,v[:,[-1]]),axis = -1)
    yaw = np.arctan2(dy,dx)
    yaw = np.concatenate((yaw,yaw[[-1]]),axis = 0 )
    return v,yaw
    
def x2a(x,y,dt):
    dy = np.diff(y,axis = -1)
    dx = np.diff(x,axis = -1)
    v = np.sqrt(dx**2+dy**2)/dt
    if np.size(np.shape(v)) == 1:
        v = np.concatenate((v,v[[-1]]),axis = 0 )
    else:
        v = np.concatenate((v,v[:,[-1]]),axis = -1)
    yaw = np.arctan2(dy,dx)
    yaw = np.concatenate((yaw,yaw[[-1]]),axis = 0 )
    
    tmpyawdif = np.append( np.diff(yaw),0)/dt
    tmpvdif = np.append(np.diff(v),0)/dt
    a = np.concatenate([np.array([tmpvdif]),np.array([tmpyawdif])],axis = 0).T

    return a
    


# def L2_dis_matrix(x_ego_list, T_start, T_end):
#     """
#     x_ego_list: list of ego_traj, first two colomn is position
#     T_start, T_end: (1,) start and end of segements 
#     """
#     n = len(x_ego_list)
#     Dis = np.zeros([n,n])
#     for i in range(n):
#         for j in range(i+1,n):
#             traj1 = x_ego_list[i][T_start:T_end,:2]
#             traj2 = x_ego_list[j][T_start:T_end,:2]
#             dis = np.sqrt( np.mean( (traj1-traj2)**2 ) )
#             Dis[i,j] = dis
#             Dis[j,i] = dis
#     return Dis
            
def L2_dis_matrix(x_ego_list, T_start, T_end, x_ego_list_2 = -1, T_start_2 = -1, T_end_2 = -1):
    """
    x_ego_list: list of ego_traj, first two colomn is position
    T_start, T_end: (1,) start and end of segements 
    """
    if x_ego_list_2 == -1:
        x_ego_list_2 = x_ego_list
    if T_start_2 == -1:
        T_start_2 = T_start
    if T_end_2 == -1:
        T_end_2 = T_end
    
    n = len(x_ego_list)
    m = len(x_ego_list_2)
    Dis = np.zeros([n,m])
    for i in np.arange(n):
        for j in np.arange(m):
            traj1 = x_ego_list[i][T_start:T_end,:2]
            traj2 = x_ego_list_2[j][T_start_2:T_end_2,:2]
            dis = np.sqrt( np.mean( (traj1-traj2)**2 ) )
            Dis[i,j] = dis
    return Dis
            
def a_dis_matrix(a_list, T_start, T_end):
    """
    a:'v,yaw'
    
    x_ego_list: list of ego_traj, first two colomn is position
    T_start, T_end: (1,) start and end of segements 
    """
    n = len(a_list)
    Dis = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1,n):
            traj1 = a_list[i][T_start:T_end,:2]
            traj2 = a_list[j][T_start:T_end,:2]
            yawdiff_1 = traj1[:,1]
            yawdiff_2 = traj2[:,1]

            dis = np.sqrt( np.mean( (np.cumsum(yawdiff_1)-np.cumsum(yawdiff_2) )**2 ) )
            
            Dis[i,j] = dis
            Dis[j,i] = dis
    return Dis
            
    
    

def sa_norm_v01(s,a):
    s_norm = 1/s
    #s_norm = ( s - 16 )/np.sqrt(78)
#    a_norm = a[:,:]/np.sqrt(0.001)
    a_norm = a[:,:]/np.sqrt([10.,0.0002])
    return s_norm,a_norm

def sa_norm_v02(s_lms,s_lmsdiff,a):
    s_norm1 = np.tanh(s_lmsdiff)
    s_norm2 = 1/s_lms
    s_norm = np.concatenate([s_norm1,s_norm2],axis = 1)
    #s_norm = ( s - 16 )/np.sqrt(78)
#    a_norm = a[:,:]/np.sqrt(0.001)
    a_norm = a[:,:]/np.sqrt([50.,0.0002])
    return s_norm,a_norm
    
def sa_norm_v03(s_lms,s_lmsdiff,s_ego,a):
    s_norm1 = np.tanh(s_lmsdiff)
    s_norm2 = 1/s_lms
    s_norm3 = s_ego[:,[3]]/25.
    s_norm = np.concatenate([s_norm1,s_norm2,s_norm3],axis = 1)
    #s_norm = ( s - 16 )/np.sqrt(78)
#    a_norm = a[:,:]/np.sqrt(0.001)
    a_norm = a[:,:]/np.sqrt([50.,0.0002])
    return s_norm,a_norm
    
def sa_norm_v04(s_lms,s_lmsdiff,s_ego,a):
    """
    for velocity action
    """
    
    if s_lms is None or s_lmsdiff is None or s_ego is None:
        s_norm = None
    else:
        
        s_norm1 = np.tanh(s_lmsdiff)
    
        s_norm2 = 1/s_lms
        s_norm3 = s_ego[:,[3]]/25.
        s_norm = np.concatenate([s_norm1,s_norm2,s_norm3],axis = 1)
        
    
    if a is not None:
        a_norm = a[:,:]/np.sqrt([400.,0.0002])
    else:
        a_norm = None
    return s_norm,a_norm


def sa_norm_v2(s_lms,s_lmsdiff,s_ego,a,s_lmsdata_pre):
    s_norm1 = np.tanh(s_lmsdiff)
    s_norm2 = 1/s_lms
    s_norm2_pre = 1/s_lmsdata_pre
    s_norm3 = s_ego[:,[3]]/25.
    a_norm = a[:,:]/np.sqrt([50.,0.0002])
    s_norm = {}
    s_norm['s_lmsdiff'] = s_norm1
    s_norm['s_lms'] = s_norm2
    s_norm['s_ego'] = s_norm3
    s_norm['s_lmsdata_pre'] = s_norm2_pre
    return s_norm,a_norm
    
def sa_norm_v202(s_lms,s_lmsdiff,s_ego,a,s_lmsdata_pre):
    """
    for velocity action
    """
    s_norm1 = np.tanh(s_lmsdiff)
    s_norm2 = 1/s_lms
    s_norm2_pre = 1/s_lmsdata_pre
    s_norm3 = s_ego[:,[3]]/25.
    a_norm = a[:,:]/np.sqrt([400.,0.0002])
    s_norm = {}
    s_norm['s_lmsdiff'] = s_norm1
    s_norm['s_lms'] = s_norm2
    s_norm['s_ego'] = s_norm3
    s_norm['s_lmsdata_pre'] = s_norm2_pre
    return s_norm,a_norm
    
    
    
    
    