#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:48:17 2018

@author: sry

use real data (traj from highwayMODT), trainset training testset test,
with a quite deal of useful updates

2-step reward succeed!
train with both max 'reward_diff' and min 's_Diff'
using velocity as action

"""
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import model
from model import reward_v02
from irl_solver_tf import *
from scipy.interpolate import interp1d
from method.method import *
from math import atan



global_save_dir = './save/demo23hu/'

action = 'v'
#action = 'acc'

if action == 'v':
    sa_norm_pointer = sa_norm_v04
    
if action == 'acc':
    sa_norm_pointer = sa_norm_v03
    
lms_single_step_gds_lines = None
def update(num,ax, navdata, line, line_forward, lms_data = None, lms_line = None, predict_data = None, predict_line = None,lms_single_step_gds = None):
    """
    Feed(update) data to the ploted vectors
    """

    ax.set_xlim(navdata[num,0]-10.0,navdata[num,0]+30.0)
    ax.set_ylim(navdata[num,1]-20.0,navdata[num,1]+20.0)
#    ax.set_ylim(navdata[0,1]-100.0,navdata[0,1]+100.0)
#    ax.set_xlim(navdata[0,0]-100.0,navdata[0,0]+100.0)
    ax.figure.canvas.draw()
    line.set_data(navdata[num,0:2])
    #line.set_linestyle('steps--')
    
    #plot lms
    if lms_data is not None:
        _,H = lms_data.shape
        lmsFrame = lms_data[num,:]
        if (H/2)*2 != H:
            lmsYaw = navdata[num,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
        else:
            lmsYaw = navdata[num,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
        lmsX = lmsFrame * np.cos(lmsYaw)  + navdata[num,0]
        lmsY = lmsFrame * np.sin(lmsYaw)  + navdata[num,1]
        lms_line.set_data( np.vstack((lmsX,lmsY)) )
        
        if lms_single_step_gds is not None:
            global lms_single_step_gds_lines
            if lms_single_step_gds_lines is not None:
                lms_single_step_gds_lines.remove()
            gds_Frame = lms_single_step_gds[num,:]
            U = gds_Frame * np.cos(lmsYaw)
            V = gds_Frame * np.sin(lmsYaw)
            lms_single_step_gds_lines = ax.quiver(lmsX,lmsY,U,V)
            
            


        
def plot_nav(nav,lmsFrame):
    plt.figure()
    plt.axis((nav[0,0]-10,nav[0,0]+30,nav[0,1]-20,nav[0,1]+20))
    plt.plot(nav[:,0],nav[:,1],'r-')
    
    H,= lmsFrame.shape
    if (H/2)*2 != H:
        lmsYaw = nav[0,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
    else:
        lmsYaw = nav[0,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
    lmsX = lmsFrame * np.cos(lmsYaw)  + nav[0,0]
    lmsY = lmsFrame * np.sin(lmsYaw)  + nav[0,1]
    plt.plot(lmsX,lmsY,'g*')   
    plt.show()
    
def plot_nav_at_T(nav,lmsFrame,T):
    plt.figure()
    plt.axis((nav[T,0]-10,nav[T,0]+30,nav[T,1]-20,nav[T,1]+20))
    plt.plot(nav[:,0],nav[:,1],'r-',label = 'ego_trajectory')
    plt.ylabel('lateral posision(m)')
    plt.xlabel('longitudinal position(m)')
    plt.plot(nav[T,0],nav[T,1],'bo')
    
    H,= lmsFrame.shape
    if (H/2)*2 != H:
        lmsYaw = nav[T,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
    else:
        lmsYaw = nav[T,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
    lmsX = lmsFrame * np.cos(lmsYaw)  + nav[T,0]
    lmsY = lmsFrame * np.sin(lmsYaw)  + nav[T,1]
    plt.grid()
    plt.plot(lmsX,lmsY,'g*',label = 'laser_point')  
    plt.legend(loc='best')
    plt.show()
    
def plot_nav_at_T_compair(nav,nav_gt,lmsFrame,T,x_ratio = 1.0, y_ratio = 1.0):
    plt.figure()
    plt.axis((nav[T,0]-10*x_ratio,nav[T,0]+30*x_ratio,nav[T,1]-20*y_ratio,nav[T,1]+20*y_ratio))
    plt.plot(nav[:,0],nav[:,1],'g-',label = 'trajectory_test')
    plt.plot(nav_gt[:,0],nav_gt[:,1],'r-', label = 'trajectory_real')

    plt.plot(nav[T,0],nav[T,1],'go')
    plt.plot(nav[T,0],nav[T,1],'ro')
#    plt.plot(nav[T,0],nav[T,1],'bo',label = 'position_test')
#    plt.plot(nav[T,0],nav[T,1],'go',label = 'position_real')
    
    H,= lmsFrame.shape
    if (H/2)*2 != H:
        lmsYaw = nav[T,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
    else:
        lmsYaw = nav[T,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
    lmsX = lmsFrame * np.cos(lmsYaw)  + nav[T,0]
    lmsY = lmsFrame * np.sin(lmsYaw)  + nav[T,1]
    plt.plot(lmsX,lmsY,'g*')   
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    
def plot_navs(nav_focus, lmsFrame = None, navs_test = None, T = 0, x_ratio = 1.0, y_ratio = 1.0):
    """
    nav:list
    ratio usage: nav[:,0]*ratiox,nav[:,1]*ratioy
    """
    plt.figure()
    plt.axis(( nav_focus[T,0]-10*x_ratio,nav_focus[T,0]+30*x_ratio,nav_focus[T,1]-20*y_ratio,nav_focus[T,1]+20*y_ratio))

    if navs_test is not None:
        for nav in navs_test:
            plt.plot(nav[:,0],nav[:,1],'g-')

    
    if lmsFrame is not None:
        H,= lmsFrame.shape
        if (H/2)*2 != H:
            lmsYaw = nav[T,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
        else:
            lmsYaw = nav[T,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
        lmsX = lmsFrame * np.cos(lmsYaw)  + nav[T,0]
        lmsY = lmsFrame * np.sin(lmsYaw)  + nav[T,1]
        plt.plot(lmsX,lmsY,'g*')   

    plt.plot(nav_focus[:,0],nav_focus[:,1],'r-',label = 'trajectory_focus')
    plt.plot(nav_focus[T,0],nav_focus[T,1],'ro')
    
        
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


#    plt.plot(np.vstack((lmsX,lmsY)),'g*') 

def visulization_0(nav,lmsFrame,reward_test,y_label = ''):
    plot_nav(nav,lmsFrame)
    plt.figure()
    plt.imshow( np.flipud(reward_test.T),cmap = 'Greys_r' )
    plt.xlabel('T')
    plt.ylabel(y_label)
    plt.title('reward')
    plt.colorbar()
    plt.show()



def draw_fake_data_v0(tmpv,tmpyaw,env_cars,dt,T,init_s = None):
    tmpyawdif = np.append( np.diff(tmpyaw),0)
    tmpvdif = np.append(np.diff(tmpv),0)
    a = np.concatenate([np.array([tmpvdif]),np.array([tmpyawdif])],axis = 0).T
    #tmpyaw = np.linspace(np.pi/2.,np.pi/2.,100)
    x_ego = v2x(tmpv,tmpyaw,dt,(0,0))
    num_of_lms_lines = 361
    lmsdata = x2lms(init_s,x_ego,env_cars,num_of_lms_lines)
    return a,x_ego,lmsdata

def draw_fake_data_v1(tmpv,tmpyaw,env_cars,dt,T,init_s = None):
    tmpyawdif = np.append( np.diff(tmpyaw),0)
    tmpvdif = np.append(np.diff(tmpv),0)
    a = np.concatenate([np.array([tmpvdif]),np.array([tmpyawdif])],axis = 0).T
    #tmpyaw = np.linspace(np.pi/2.,np.pi/2.,100)
    x_ego = v2x(tmpv,tmpyaw,dt,(0,0))
    num_of_lms_lines = 361
    lmsdata = x2lms(init_s,x_ego,env_cars,num_of_lms_lines)
    return a,x_ego,lmsdata    

def traj2env(traj,dt):
    env_cars_x = traj[:,:,1]
    env_cars_y = traj[:,:,2]
    _,env_cars_yaw = x2yaw(env_cars_x,env_cars_y,dt)
    env_cars = np.array([env_cars_x,env_cars_y,env_cars_yaw])
    return env_cars
    
dir_path = '/home/hsc/Code/IV2019/SunCode/data_lane_change/lane_pose_0/'




nav_list = []
nav_list_test = []
traj_list = []
traj_list_test = []

#bug data:212
#216:rear car influence

#np.arange(1,282)
for i in np.arange(213,288):
    navfile = dir_path + 'nav%d'%i
    trajfile = dir_path + 'traj%d'%i
    nav = sio.loadmat(navfile)
    nav = nav['nav_list']
    nav_list.append(nav)
    traj = sio.loadmat(trajfile)
    traj = traj['new_traj']
    traj_list.append(traj)

    
for i in np.arange(100,211):
    navfile = dir_path + 'nav%d'%i
    trajfile = dir_path + 'traj%d'%i
    nav = sio.loadmat(navfile)
    nav = nav['nav_list']
    nav_list.append(nav)
    traj = sio.loadmat(trajfile)
    traj = traj['new_traj']
    traj_list.append(traj)    

#for i in np.arange(1,100):
#    navfile = dir_path + 'nav%d'%i
#    trajfile = dir_path + 'traj%d'%i
#    nav = sio.loadmat(navfile)
#    nav = nav['nav_list']
#    nav_list.append(nav)
#    traj = sio.loadmat(trajfile)
#    traj = traj['new_traj']
#    traj_list.append(traj)

init_s = None
#init_s = {}
#init_s['lw'] = 12.0 
#init_s['rw'] = -2.0
dt = 0.1
num_of_lms_lines = 361

def process_Data_real(cur_nav,cur_traj,dt):
    """
    return: x_ego,lmsdata,lmsdata_diff,a,lmsdata_pre
    """
    cur_traj = np.transpose(cur_traj,axes = [1,0,2])
    #traj:milli, fno, gp.x, gp.y, glen0, glen1, gv1.x, gv1.y, ep.x, ep.y, ev1.x, ev1.y, interfrmspd
    tmpv,tmpyaw = x2yaw(cur_nav[:,1],cur_nav[:,2],dt)
    tmpyawdif = np.append( np.diff(tmpyaw),0)/dt
    tmpvdif = np.append(np.diff(tmpv),0)/dt
    a = np.concatenate([np.array([tmpvdif]),np.array([tmpyawdif])],axis = 0).T
    x_ego1 = np.concatenate([cur_nav[:,[1]],cur_nav[:,[2]],np.expand_dims(tmpyaw,axis = 1),np.expand_dims(tmpv,axis = 1)],axis = 1)
    env_cars = np.concatenate( (cur_traj[:,:,1:3],np.zeros_like(cur_traj[:,:,[1]]) ), axis = 2)
    lmsdata1 = x2lms(init_s,x_ego1,env_cars,num_of_lms_lines)
    lmsdata1_diff = np.concatenate([ [np.zeros(num_of_lms_lines)], lmsdata1[1:,]-lmsdata1[:-1,] ],axis = 0)
    return x_ego1[1:-2,:],lmsdata1[1:-2,:],lmsdata1_diff[1:-2,:],a[1:-2,:],lmsdata1[:-3,:]

def process_Data_real_velocityAction(cur_nav,cur_traj,dt):
    """
    return: x_ego,lmsdata,lmsdata_diff,a,lmsdata_pre
    """
    cur_traj = np.transpose(cur_traj,axes = [1,0,2])
    #traj:milli, fno, gp.x, gp.y, glen0, glen1, gv1.x, gv1.y, ep.x, ep.y, ev1.x, ev1.y, interfrmspd
    tmpv,tmpyaw = x2yaw(cur_nav[:,1],cur_nav[:,2],dt)
    tmpyawdif = np.append( np.diff(tmpyaw),0)/dt
#    tmpvdif = np.append(np.diff(tmpv),0)/dt
    a = np.concatenate([np.array([tmpv]),np.array([tmpyawdif])],axis = 0).T
    x_ego1 = np.concatenate([cur_nav[:,[1]],cur_nav[:,[2]],np.expand_dims(tmpyaw,axis = 1),np.expand_dims(tmpv,axis = 1)],axis = 1)
    env_cars = np.concatenate( (cur_traj[:,:,1:3],np.zeros_like(cur_traj[:,:,[1]]) ), axis = 2)
    lmsdata1 = x2lms(init_s,x_ego1,env_cars,num_of_lms_lines)
    lmsdata1_diff = np.concatenate([ [np.zeros(num_of_lms_lines)], lmsdata1[1:,]-lmsdata1[:-1,] ],axis = 0)
    return x_ego1[1:-2,:],lmsdata1[1:-2,:],lmsdata1_diff[1:-2,:],a[1:-2,:],lmsdata1[:-3,:]



#x_ego = np.array([x_ego1,x_ego2,x_ego3,x_ego4,x_ego5])
#lmsdata = np.array([lmsdata1,lmsdata2,lmsdata3,lmsdata4,lmsdata5])
#lmsdata_diff = np.array([lmsdata1_diff,lmsdata2_diff,lmsdata3_diff,lmsdata4_diff,lmsdata5_diff])
#a = np.array([a1,a2,a3,a4,a5])


#x_ego = np.array([x_ego1,x_ego2,x_ego3])
#lmsdata = np.array([lmsdata1,lmsdata2,lmsdata3])
#lmsdata_diff = np.array([lmsdata1_diff,lmsdata2_diff,lmsdata3_diff])
#a = np.array([a1,a2,a3])



#x_ego = [x_ego1,x_ego2]
#lmsdata = [lmsdata1,lmsdata2]
#lmsdata_diff = [lmsdata1_diff,lmsdata2_diff]
#a = [a1,a2]



def process_Data_real_all(nav_list,traj_list,dt):
    x_ego = []
    lmsdata = []
    lmsdata_diff = []
    a = []
    lmsdata_pre = []
    l = np.size(nav_list)
    for i in range(l):
        cur_nav = nav_list[i]
        cur_traj = traj_list[i]
        x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
        x_ego.append(x_ego1)
        lmsdata.append(lmsdata1)
        lmsdata_diff.append(lmsdata1_diff)
        a.append(a1)
        lmsdata_pre.append(lmsdata1_pre)
    return x_ego,lmsdata,lmsdata_diff,a,lmsdata_pre

def process_Data_real_velocityAction_all(nav_list,traj_list,dt):
    x_ego = []
    lmsdata = []
    lmsdata_diff = []
    a = []
    lmsdata_pre = []
    l = np.size(nav_list)
    for i in range(l):
        cur_nav = nav_list[i]
        cur_traj = traj_list[i]
        x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real_velocityAction(cur_nav,cur_traj,dt)
        x_ego.append(x_ego1)
        lmsdata.append(lmsdata1)
        lmsdata_diff.append(lmsdata1_diff)
        a.append(a1)
        lmsdata_pre.append(lmsdata1_pre)
    return x_ego,lmsdata,lmsdata_diff,a,lmsdata_pre

if action == 'v':
    process_Data_real = process_Data_real_velocityAction
    process_Data_real_all = process_Data_real_velocityAction_all
    
x_ego,lmsdata,lmsdata_diff,a_list,lmsdata_pre = process_Data_real_all(nav_list[11:],traj_list[11:],dt)
#x_ego,lmsdata,lmsdata_diff,a = process_Data_real_all(nav_list,traj_list,dt)

cur_nav = nav_list[0]#left lane change
cur_traj = traj_list[0]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
    
cur_nav = nav_list[10]#right lane change
cur_traj = traj_list[10]
x_ego2,lmsdata2,lmsdata2_diff,a2,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)


def traj_animation(x_ego2,lmsdata2,lms_single_step_gds = None):
    
    Tnav,_ = np.shape(x_ego2)
    Tlms,_ = np.shape(lmsdata2)
    if lms_single_step_gds is not None:
        Tgds,_ = np.shape(lms_single_step_gds)
        T = min(Tnav,Tlms,Tgds)
    else:
        T = min(Tnav,Tlms)
    
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)

    
    l, = ax.plot([], [], 'ro')#nav
    j, = ax.plot([], [], 'g*')#lms
    k, = ax.plot([], [], 'r-')#predict_previous
    g  = ax.quiver([], [],[],[],pivot='mid', color='r', units='inches')#single_step_gds
    t, = ax.plot([], [], 'g+')#nav_after
    
    
    ax.grid()
    #plt.xlim(0, 2000)
    #plt.ylim(-1000, 1000)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1,update, T, fargs=(ax,x_ego2, l, t, lmsdata2, j, None, k, lms_single_step_gds),
                                       interval=50, blit=False)
    plt.show()
    
#traj_animation(x_ego2,lmsdata2)

"""
v0.1
"""

def reshape_for_single_step(data_list):
    return np.concatenate(data_list,axis = 0)
    
def reshape_for_M_step(data_list,M):
    """
    M: memory_frame
    """
    ind_hash = np.array([],dtype = np.int32)
    ind_pre = 0
    for data in data_list:
        n,d = np.shape(data)
        ind = np.arange(n-M+1,dtype = np.int32)+ind_pre
        ind_pre = ind_pre + n
        ind_hash = np.concatenate([ind_hash,ind])
        
        
    return np.concatenate(data_list,axis = 0),ind_hash

def reshape_for_test_traj_multi(data_list,T_start = 0):
    """
    T_start: to choose from which timestep the test begin
    """
    ind_hash = np.array([],dtype = np.int32)
    ind_pre = 0
    for data in data_list:
        n,d = np.shape(data)
        ind = ind_pre + T_start
        ind_pre = ind_pre + n
        ind_hash = np.concatenate([ind_hash,[ind]])
        
        
    return np.concatenate(data_list,axis = 0),ind_hash

    

    
    
#def sa_norm_pointer(s_lms,s_lmsdiff,????,a):
#    s_norm1 = np.tanh(s_lmsdiff)
#    s_norm2 = 1/s_lms
#    s_norm = np.concatenate([s_norm1,s_norm2],axis = 1)
#    #s_norm = ( s - 16 )/np.sqrt(78)
##    a_norm = a[:,:]/np.sqrt(0.001)
#    a_norm = a[:,:]/np.sqrt([1.,0.001])
#    return s_norm,a_norm
#    


def KDE(samples,bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples)
    return  np.exp(kde.score_samples(samples))

def knn_density(samples,k):
    """
    samples:(N,D)
    return the 'unnormalized' density on each samples
    """
    N,D = np.shape(samples)
    neigh = NearestNeighbors(metric = 'euclidean')
    neigh.fit(samples)
    a = neigh.kneighbors(samples, k, return_distance=True)
    radius = np.max(a[0],axis = 1)
    d = 1/(radius**D)#will cause dimension 
    return d
    
def knn_density_2D(samples,k):
    """
    samples:(N,D)
    return the 'unnormalized' density on each samples
    """
    N,D = np.shape(samples)
    neigh = NearestNeighbors(metric = 'euclidean')
    neigh.fit(samples)
    a = neigh.kneighbors(samples, k, return_distance=True)
    radius = np.max(a[0],axis = 1)
    d = 1/(radius**2)#sorry
    return d
    
def creating_index_matrix_without_ind(start_line,memory_frame):
    """
    start_line:(N,)
    """        
    N, = np.shape(start_line)
    batch_index = np.expand_dims(start_line,1)+np.arange(memory_frame)
    return batch_index

"""
Below is a demo: how to use KDE to estimate dentisy and generate samples
"""
"""
data_a = a_norm[creating_index_matrix_without_ind(ind_hash,M)]#(-1,M,D)
                
data_a = np.reshape(data_a,[-1,2*M])
kde = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_a)
kde2 = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_a)

samples = kde.sample(10000)
scores = kde2.score_samples(samples)

samples = np.reshape(samples,[-1,M,2])
data_a = np.reshape(data_a,[-1,M,2])

#tmpx = np.arange(-2,2,0.01)
#tmpy = np.arange(-2,2,0.01)
#xx,yy = np.meshgrid(tmpx,tmpy)
#res = []
#for i in tmpx:
#    for j in tmpy:
#        cur_res = np.exp(kde.score_samples(np.array([[i,j]])))
#        res.append(cur_res)
#        
#res = np.reshape(np.array(res),[400,400])
#plt.contour(xx,yy,res)
#plt.colorbar()
#plt.show()

dim = 1
range_min = [[-0.5,0.5],[-0.5,0.5]]
range_max = [[-2,2],[-2,2]]

fig = plt.hist2d(samples[:,0,dim], samples[:,1,dim], bins=400, range = range_max)
plt.set_cmap('jet')
cb = plt.colorbar()
cb.set_label('counts')
plt.show()
#plt.plot(data_a[:,0],data_a[:,1],'o')
fig = plt.hist2d(data_a[:,0,dim], data_a[:,1,dim], bins=400,range = range_max)
plt.set_cmap('jet')
cb = plt.colorbar()
cb.set_label('counts')
plt.show()
"""

"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""

M = 20
test_train_partition = 80

learning_rate_list = [0.5,0.5,0.5,0.5,
                      0.5,0.5,0.5,0.5,
                      0.1,0.1,0.1,0.1,
                      0.1,0.02,0.02,0.02,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01,0.01,0.01,0.01,
                      0.01]

                      
#s_ego,s_ego_hash = reshape_for_M_step(x_ego[test_train_partition:],M)
#s_lms,s_lms_hash = reshape_for_M_step(lmsdata[test_train_partition:],M)
#s_lmsdiff,s_lmsdiff_hash = reshape_for_M_step(lmsdata_diff[test_train_partition:],M)
#s_lmsdata_pre,s_lmsdata_pre_hash = reshape_for_M_step(lmsdata_pre[test_train_partition:],M)
#a,a_hash = reshape_for_M_step(a_list[test_train_partition:],M)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@

np.random.seed(0)
hsc_traj_list = []
hsc_xy_list = []
num = 100
start = 10000

#d = np.random.randint(0,40000,size=(100,1))
#for i in d:
#    x,y = np.loadtxt('/home/sry/new_story/trajGenerate/Trajs/traj%g.txt'%(i))
#    hsc_xy_list.append(np.array([x,y]).T)
#    one_traj = np.zeros([len(x)-1,2])
#    for j in range(1,len(x)):
#        one_traj[j-1,0] = pow(pow((x[j]-x[j-1]),2)+pow((y[j]-y[j-1]),2),0.5)*10
#        if (j>1):
#            one_traj[j-1,1] = atan((y[j]-y[j-1])/(x[j]-x[j-1]))-atan((y[j-1]-y[j-2])/(x[j-1]-x[j-2]))
#        else:
#            one_traj[0,1] = atan((y[j]-y[j-1])/(x[j]-x[j-1]))
#            
#    hsc_traj_list.append(one_traj)
#


trajGen_range = np.linspace(1,175,175)
np.random.seed(0)
for i in trajGen_range:
    for k in np.linspace(0,26,27):
#    for k in [13]:
        x,y= np.loadtxt('/home/hsc/Code/IV2019/5poly_Trajs/5poly_traj%g_%g.txt'%(i,k))
        hsc_xy_list.append(np.array([x,y]).T)
        tmpv,tmpyaw = x2yaw(x,y,dt)
        tmpyawdif = np.append( np.diff(tmpyaw),0)/dt
#    tmpvdif = np.append(np.diff(tmpv),0)/dt
        one_traj = np.concatenate([np.array([tmpv]),np.array([tmpyawdif])],axis = 0).T
#        if np.random.rand() > 0.9:
        hsc_traj_list.append(one_traj)

"""
save out the trajectories
"""
#plt.figure()
#cnt = 1
#for x_ego_xy in x_ego:
#    plt.plot(x_ego_xy[:,0],x_ego_xy[:,1],'r-')#real trajectories
#    x_tmp = x_ego_xy[:,0]
#    y_tmp = x_ego_xy[:,1]
#    size_tmp = x_tmp.size
#    x_t = x_tmp[50:size_tmp-1-50]
#    y_t = y_tmp[50:size_tmp-1-50]
#    np.savetxt("/home/sry/new_story/trajGenerate/TrajFromSunCode/traj%g.txt"%(cnt),[x_t,y_t])
#    cnt = cnt+1
"""
end
"""

plt.figure()
for x_ego_xy in x_ego:
    plt.plot(x_ego_xy[:,0],x_ego_xy[:,1],'r-')#real trajectories

for hsc_xy in hsc_xy_list:
    plt.plot(hsc_xy[:,0],hsc_xy[:,1],'g-')#generated trajectories


    
        
        
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ plot traj.




a_norm_policy = None
a_policy_hash = None


"""
For Hu: traj. generator @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""


a_list_policy  = hsc_traj_list #??? your generated trajectory
#a_list_policy: should be a list, each component in the list should be of shape (length_of_trajectory, 2).
#for example:
#   [(173, 2),
#   (185, 2),
#   (177, 2),
#    ...    ]
THE_START_FRAME_OF_LANE_CHANGE = 0# you can use 'THE_START_OF_LANE_CHANGE = 0' if the start point in your trajectory is frame 0
a_policy,a_policy_hash = reshape_for_test_traj_multi(a_list_policy,THE_START_FRAME_OF_LANE_CHANGE)
_,a_norm_policy = sa_norm_pointer(None,None,None,a_policy)



"""
end
"""


s_ego,s_ego_hash = reshape_for_test_traj_multi(x_ego[test_train_partition:],48)
s_lms,s_lms_hash = reshape_for_test_traj_multi(lmsdata[test_train_partition:],48)
s_lmsdiff,s_lmsdiff_hash = reshape_for_test_traj_multi(lmsdata_diff[test_train_partition:],48)
s_lmsdata_pre,s_lmsdata_pre_hash = reshape_for_test_traj_multi(lmsdata_pre[test_train_partition:],48)
a,a_hash = reshape_for_test_traj_multi(a_list[test_train_partition:],48)


s_norm,a_norm = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
ind_hash = s_ego_hash


mode = 'train'
#'load'
load_train = False




if mode == 'train':
    with tf.Graph().as_default():#creat a new clean graph
        solver   = IRL_Solver_demo21(s_norm, a_norm, ind_hash, a_norm_policy, a_policy_hash,
                   start_point_range = 1,  # selected the start point of training sequence should < this   
                   update_rule='sgd',
                   batch_size = min(1024,ind_hash.shape[0]),
                   num_of_lms_lines = num_of_lms_lines,
    #               hidden_size = 100,
                   iteration=1000,
                   memory_frame = M,
                   optim_config={
                     'learning_rate': learning_rate_list[M],#0:creat a network with ini_state
                     'decay':1
                   },
                   regularizer_scale = 1e-6,
                   print_every=10,
                   depict = 'irl',
                   test_only = False,
                   load_train = load_train,
                   save_dir = global_save_dir,
                   sample_expert_ratio = 9,
                   sampling_range_list = np.array([[0,0],[1.,8.]]),
                   )
        solver.train()
    
    
    
    plt.plot(solver.reward_diff_history,label = 'reward_diff')
    plt.plot(solver.reward_expert_history,label = 'reward_expert')
    plt.plot(solver.reward_policy_history,label = 'reward_policy')
    plt.legend(loc='best')
    plt.title('rward history')
    #plt.savefig('%s/train_loss.jpg' % ann_solver2.save_dir)
    plt.show()

"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""
#


#def grid_search3d(x,y,z):
#    """
#    warning: low performance
#    """
#    for i in x:
#        for j in y:
#            for k in z:
#                yield (i,j,k)

#x_test = np.linspace(-10,30,41)
#y_test = np.linspace(-1.8,11.8,15)
#yaw_test = np.linspace(-np.pi/50,np.pi/50,21)
#lx, = np.shape(x_test)
#ly, = np.shape(y_test)
#lyaw, = np.shape(yaw_test)     
           
#lms_test = []
#a_test = []
#env_cars = np.tile(np.array([[[10,6,0],[25,1,0],[15,7,0.2],[4,9,-0.1],[34,2,0.06]]]),[T,1,1])
#for a_data in grid_search3d(x_test,y_test,yaw_test):
#    lms_test.append(creat_lms_frame(init_s,a_data,env_cars[0,:,:],num_of_lms_lines)[:num_of_lms_lines])
#    a_test.append([a_data[2]])
#s = np.array(lms_test)
#a = np.array(a_test)
#
#s_norm_new = ( s - 16 )/np.sqrt(78)
#a_norm_new = a[:,[0]]/np.sqrt(0.001)
#traj_animation(x_ego1,lmsdata1,single_step_gds_decode)
#with tf.Graph().as_default():#creat a new clean graph
#    solver_test   = IRL_Solver(s_norm_new, a_norm_new,
#               start_point_range = 1,  # selected the start point of training sequence should < this   
#               update_rule='sgd',
#               num_epochs=0,
#               batch_size=32,
#               optim_config={
#                 'learning_rate': 3e-5,#0:creat a network with ini_state
#               },
#               depict = 'irl',
#               test_only = True,
#               print_every = 1000
#               )
#    solver_test.train()
#reward_test = np.array(solver_test.reward_expert_history)
#reward_test = np.reshape(reward_test,(lx*ly,lyaw))
#arg_max_reward = np.argmax(reward_test,axis = 1)
#arg_max_reward = np.reshape(arg_max_reward,(lx,ly))
#

def grid_search2d(xy,z):
    """
    warning: low performance
    """
    for ij in xrange(np.shape(xy)[0]):
            for k in z:
                yield (xy[ij],k)
                
def grid_search3d(x,y,z):
    """
    warning: low performance
    """
    for i in xrange(np.shape(x)[0]):    
        for j in xrange(np.shape(y)[0]):
            for k in xrange(np.shape(z)[0]):
                yield (x[i],y[j],z[k])
                
def creat_testing_data_v02(lmsdata_test_list,lmsdata_test_diff_list,x_ego_list,yaw_test,vdiff_test):
    """
    only for single-step
    """
    lms_test = []
    lms_test_diff = []
    a_test = []
    x_ego_test = []
    for step,lmsdata_test in enumerate(lmsdata_test_list):
        lmsdata_test_diff = lmsdata_test_diff_list[step]
        xego = x_ego_list[step]
        for data in grid_search3d(lmsdata_test,yaw_test,vdiff_test):
            lms_test.append(data[0])
            a_test.append([data[2],data[1]])
        for data in grid_search3d(lmsdata_test_diff,yaw_test,vdiff_test):
            lms_test_diff.append(data[0])
        for data in grid_search3d(xego,yaw_test,vdiff_test):
            x_ego_test.append(data[0])
    s_lms = np.array(lms_test)
    s_lmsdiff = np.array(lms_test_diff)
    s_ego = np.array(x_ego_test)
    a = np.array(a_test)
    ind_hash = np.arange(np.shape(a)[0],dtype = np.int32)
    return s_lms,s_lmsdiff,s_ego,a, ind_hash
                


yaw_test_range = [np.pi/64,21]
vdiff_test_range = [10.*dt,21]
v_test_range = [12.,24.,21]

if action == 'v':
    vdiff_test_range = v_test_range
lmsdata_list = [lmsdata1,lmsdata2]
x_ego_list = [x_ego1,x_ego2]
lmsdata_diff_list = [lmsdata1_diff,lmsdata2_diff]

"""
for debuging test_one_traj(...)
cur_nav = nav_list[0]#left lane change
cur_traj = traj_list[10]#right lane change
cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
s_lms = lmsdata1
s_lmsdiff = lmsdata1_diff
s_ego = x_ego1
a = a1
"""



def test_one_traj(s_lms,s_lmsdiff,a,s_ego,save_dir):
    """
    single-step
    """
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    ind_hash = np.arange(np.shape(a)[0],dtype = np.int32)

    with tf.Graph().as_default():#creat a new clean graph
            solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                       start_point_range = 1,  # selected the start point of training sequence should < this   
                       update_rule='sgd',
                       memory_frame = 1,
                       num_of_lms_lines = num_of_lms_lines,
        #               hidden_size = 100,
                       num_epochs=0,
                       optim_config={
                         'learning_rate': 3e-5,#0:creat a network with ini_state
                       },
                       depict = 'irl',
                       test_only = True,
                       print_every = 1000,
                       save_dir = save_dir,
                       sampling_range_list = np.array([[0,0],[1.,8.]])
                       )
            solver_test.train()    
    reward_test = np.array(solver_test.reward_expert_history)
    plt.figure()
    plt.ylabel('reward')
    plt.xlabel('T(0.1s)')
    plt.axis([0,140,-2,2])
    plt.plot(reward_test)
    plt.show()
    return reward_test
    
def test_one_traj_multi_step(s_lms,s_lmsdiff,a,s_ego,save_dir,M_test = 100):
    """
    multi-step
    """
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    ind_hash = np.arange(1,dtype = np.int32)

    with tf.Graph().as_default():#creat a new clean graph
            solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                       start_point_range = 1,  # selected the start point of training sequence should < this   
                       update_rule='sgd',
                       memory_frame = M_test,
                       num_of_lms_lines = num_of_lms_lines,
        #               hidden_size = 100,
                       batch_size = 1,#can remove(removed: duplicated calulation but no bugs)
                       num_epochs=0,
                       optim_config={
                         'learning_rate': 3e-5,#0:creat a network with ini_state
                       },
                       depict = 'irl',
                       test_only = True,
                       print_every = 1000,
                       save_dir = save_dir,
                       sampling_range_list = np.array([[0,0],[1.,8.]])
                       )
            solver_test.train()    
    reward_test = np.array(solver_test.reward_expert_history)
    return reward_test
    


def test_multi_traj_multi_step(lmsdata,x_ego,lmsdata_diff,lmsdata_pre,a,save_dir,T_start = 0, M_test = 100):
    """
    multi-step,multi-traj,batch version of func:'test_one_traj_multi_step'
    - lmsdata,x_ego,lmsdata_diff,lmsdata_pre,a: list
    """
    s_ego,s_ego_hash = reshape_for_test_traj_multi(x_ego,T_start)
    s_lms,s_lms_hash = reshape_for_test_traj_multi(lmsdata,T_start)
    s_lmsdiff,s_lmsdiff_hash = reshape_for_test_traj_multi(lmsdata_diff,T_start)
    s_lmsdata_pre,s_lmsdata_pre_hash = reshape_for_test_traj_multi(lmsdata_pre,T_start)
    #s = np.concatenate([s_lms,s_ego],axis = 1)
    a,a_hash = reshape_for_test_traj_multi(a,T_start)
    
    
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    ind_hash = s_ego_hash
    
    batch_size = np.shape(ind_hash)[0]

    with tf.Graph().as_default():#creat a new clean graph
            solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                       start_point_range = 1,  # selected the start point of training sequence should < this   
                       update_rule='sgd',
                       memory_frame = M_test,
                       num_of_lms_lines = num_of_lms_lines,
        #               hidden_size = 100,
                       batch_size = batch_size,#can remove(removed: duplicated calulation but no bugs)
                       num_epochs=0,
                       optim_config={
                         'learning_rate': 3e-5,#0:creat a network with ini_state
                       },
                       depict = 'irl',
                       test_only = True,
                       print_every = 1000,
                       save_dir = save_dir,
                       sampling_range_list = np.array([[0,0],[1.,8.]])
                       )
            solver_test.train()    
    reward_test = np.array(solver_test.reward_expert_history)
    return reward_test



def s_a_cross_test_multi_traj_multi_step(lmsdata,x_ego,lmsdata_diff,lmsdata_pre,a_list_test,save_dir,T_start_s = 0,T_start_a = 0, M_test = 100):
    """
    multi-step,multi-traj,batch version of func:'test_one_traj_multi_step'
    - lmsdata,x_ego,lmsdata_diff,lmsdata_pre,a: list
    
    batch on 's', circle on 'a'
    """
    
    s_ego,s_ego_hash = reshape_for_test_traj_multi(x_ego,T_start_s)
    s_lms,s_lms_hash = reshape_for_test_traj_multi(lmsdata,T_start_s)
    s_lmsdiff,s_lmsdiff_hash = reshape_for_test_traj_multi(lmsdata_diff,T_start_s)
    s_lmsdata_pre,s_lmsdata_pre_hash = reshape_for_test_traj_multi(lmsdata_pre,T_start_s)
    #s = np.concatenate([s_lms,s_ego],axis = 1)
    a,a_hash = reshape_for_test_traj_multi(a_list_test,T_start_a)
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    n = np.shape(x_ego)[0]
    m = np.shape(a_list_test)[0]
    
    s_a_cross_reward = np.zeros([m,n])
    ind_hash = s_ego_hash
    
    with tf.Graph().as_default():#creat a new clean graph
        solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                   start_point_range = 1,  # selected the start point of training sequence should < this   
                   update_rule='sgd',
                   memory_frame = M_test,
                   num_of_lms_lines = num_of_lms_lines,
    #               hidden_size = 100,
                   batch_size = np.shape(ind_hash)[0],#can remove(removed: duplicated calulation but no bugs)
                   num_epochs=0,
                   optim_config={
                     'learning_rate': 3e-5,#0:creat a network with ini_state
                   },
                   depict = 'irl',
                   test_only = True,
                   print_every = 1000,
                   save_dir = save_dir,
                   sampling_range_list = np.array([[0,0],[1.,8.]]),
                   )
        sess = solver_test.prepare_sess()
        for cnt,i_a in enumerate(a_hash):
            print ('processing trajectory:',cnt)
            a_hash_same_traj = np.zeros_like(a_hash) + i_a
            solver_test.a_hash = a_hash_same_traj
            solver_test._reset()
            solver_test._step(sess = sess)    
            reward_test = np.reshape(np.array(solver_test.reward_expert_history),[-1])
            solver_test.reward_expert_history = []
            s_a_cross_reward[cnt,:] = reward_test
        sess.close()

    
    return s_a_cross_reward



def visulization_1(nav,lmsFrame,reward_test,gt,y_label = '',save_dir = '.'):
    plot_nav(nav,lmsFrame)
    plt.figure()
    plt.imshow( np.flipud(reward_test.T),cmap = 'Greys_r' )
    plt.plot(gt)
    plt.xlabel('T(0.1s)')
    plt.ylabel(y_label)
    plt.title('reward')
    plt.colorbar()
    plt.savefig('%s/%s.png' % (save_dir,y_label))
    plt.show()

def QAQ(yaw_test_range,vdiff_test_range,lmsdata_list,lmsdata_diff_list,a_gt,x_ego_list,save_dir):
    
    yaw_test = np.linspace(-yaw_test_range[0],yaw_test_range[0],yaw_test_range[1])#0.05
    if len(vdiff_test_range) == 2:
        vdiff_test = np.linspace(-vdiff_test_range[0],vdiff_test_range[0],vdiff_test_range[1])
    if len(vdiff_test_range) == 3:
        vdiff_test = np.linspace(vdiff_test_range[0],vdiff_test_range[1],vdiff_test_range[2])
    lyaw, = np.shape(yaw_test)
    lv, = np.shape(vdiff_test)

     
#s_lms,s_lmsdiff,a = creat_testing_data_v01([lmsdata4,lmsdata5],[lmsdata4_diff,lmsdata5_diff],yaw_test,vdiff_test)
#s_lms,s_lmsdiff,a = creat_testing_data_v01([lmsdata1,lmsdata2,lmsdata3,lmsdata4,lmsdata5],[lmsdata1_diff,lmsdata2_diff,lmsdata3_diff,lmsdata4_diff,lmsdata5_diff],yaw_test,vdiff_test)
    s_lms,s_lmsdiff,s_ego,a,ind_hash = creat_testing_data_v02(lmsdata_list,lmsdata_diff_list,x_ego_list,yaw_test,vdiff_test)


    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)

#a_norm_gt = a_gt[:,[1]]/np.sqrt(0.001)


    
    def gt_scale2label_scale(raw,test_range):
        if len(test_range) == 2:
            middle = np.floor(test_range[1]/2 )
            if np.mod(test_range[1],2) == 0:
                raise ValueError('test_range[1] must be odds')
            res = test_range[0]*2/ (test_range[1]-1)
            idx_scale = -raw/res + middle
            
        if len(test_range) == 3:
            """
            left, right, num_of_label
            """
            middle = np.floor(test_range[2]/2 )
            if np.mod(test_range[2],2) == 0:
                raise ValueError('test_range[1] must be odds')
            res = (test_range[1] - test_range[0]) / (test_range[2]-1)
            middle_gt_scale = np.float(test_range[1] + test_range[0])/2
            idx_scale = -(raw-middle_gt_scale)/res + middle           
            
        return idx_scale
        
    
    with tf.Graph().as_default():#creat a new clean graph
        solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                   start_point_range = 1,  # selected the start point of training sequence should < this   
                   update_rule='sgd',
                   memory_frame = 1,
                   num_of_lms_lines = num_of_lms_lines,
    #               hidden_size = 100,
                   num_epochs=0,
                   optim_config={
                     'learning_rate': 3e-5,#0:creat a network with ini_state
                   },
                   depict = 'irl',
                   test_only = True,
                   print_every = 1000,
                   save_dir = global_save_dir,
                   sampling_range_list = np.array([[0,0],[1.,8.]])
                   )
        solver_test.train()
    reward_test = np.array(solver_test.reward_expert_history)
    reward_test = np.reshape(reward_test,(-1,lyaw,lv))
    #reward_test1 = reward_test[:,10,:]
    reward_test1 = reward_test[:np.shape(lmsdata1)[0],:,10]#10 means the middle
    arg_max_reward = np.argmax(reward_test1,axis = 1)
    a_gtyawlabel = gt_scale2label_scale(a_gt[:,1],yaw_test_range)
    visulization_1(x_ego1,lmsdata1[0,:],reward_test1,a_gtyawlabel,y_label = 'steering angle(index)',save_dir = save_dir)
    plt.plot( arg_max_reward,'o')
    
    
    
    reward_test1 = reward_test[:np.shape(lmsdata1)[0],10,:]
    arg_max_reward = np.argmax(reward_test1,axis = 1)
    a_gtvlabel = gt_scale2label_scale(a_gt[:,0],vdiff_test_range)
    visulization_1(x_ego1,lmsdata1[0,:],reward_test1,a_gtvlabel,y_label = 'v(index)', save_dir = save_dir)
    plt.plot( arg_max_reward,'o')
  
#QAQ(yaw_test_range,vdiff_test_range,lmsdata_list,lmsdata_diff_list)
"""
's-a-cross' test
"""
def s_a_cross_test(yaw_test_range,vdiff_test_range,cur_nav,cur_traj,dt,depict = ''):
    
    save_dir = '%s/%s/' % (global_save_dir,depict)
    try:
        os.mkdir(save_dir)
    except:
        pass

    T_nav,_ = np.shape(cur_nav)
    _,T_traj,_ = np.shape(cur_traj)
    T = min(T_nav,T_traj)
    cur_nav = cur_nav[:T]
    cur_traj = cur_traj[:T]
    x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
    traj_animation(x_ego1,lmsdata1)
    QAQ(yaw_test_range,vdiff_test_range,[lmsdata1],[lmsdata1_diff],a1,[x_ego1],save_dir = save_dir)
    


    
    
def thesis_plot(nav,lmsFrame,reward_test,gt,y_label = '',save_dir = '.'):
    plot_nav(nav,lmsFrame)
    plt.figure()
    plt.imshow( np.flipud(reward_test.T),cmap = 'Greys_r' )
    plt.plot(gt)
    plt.xlabel('T')
    plt.ylabel(y_label)
    plt.title('reward')
    plt.colorbar()
    plt.savefig('%s/%s.png' % (save_dir,y_label))
    plt.show()

T = 150

"""
# criterion 1 """
x_ego_test = x_ego[:test_train_partition]
a_list_test = a_list[:test_train_partition]
lmsdata_test = lmsdata[:test_train_partition]
lmsdata_diff_test = lmsdata_diff[:test_train_partition]
lmsdata_pre_test = lmsdata_pre[:test_train_partition]

#L2_dis_matrix is modified
Dis_a = L2_dis_matrix(a_list_test,48,77)#compute dis of 'action'
Dis_x = L2_dis_matrix(x_ego_test,48,77)

#plt.scatter(np.reshape(Dis_a,[-1]),np.reshape(Dis_x,[-1]))


#reward_all = test_multi_traj_multi_step(lmsdata,x_ego,lmsdata_diff,lmsdata_pre,a_list,global_save_dir,T_start = 48, M_test = 50)

#return value: row -> action, column -> state
s_a_cross_reward_all = s_a_cross_test_multi_traj_multi_step(lmsdata_test,x_ego_test,lmsdata_diff_test,lmsdata_pre_test,a_list_test,global_save_dir,T_start_s = 48,T_start_a = 48, M_test = 20)
#s_a_cross_reward_all  = s_a_cross_reward_all/np.mean(s_a_cross_reward_all,axis = 0,keepdims = True)
plt.figure()
plt.imshow(s_a_cross_reward_all,cmap = 'Greys_r')
plt.colorbar()
plt.show()

def dist_reward_plot_for_matrix(Dis,bin_num,s_a_cross_reward_all):
    gap = np.max(Dis + 0.0001)/bin_num
    bins = np.zeros(bin_num)
    traj_per_bin = np.zeros(bin_num)
    n,m = Dis.shape
    point = []
    for i in xrange(n):
        for j in xrange(m):
            idx = np.int(np.floor(Dis[i,j]/gap))
            bins[idx] = bins[idx] + s_a_cross_reward_all[i,j]
            traj_per_bin[idx] = traj_per_bin[idx] + 1
            point.append((Dis[i,j],s_a_cross_reward_all[i,j]))
    point = np.array(point)
    bins = bins/traj_per_bin
    plt.figure()
    plt.plot(bins,'o')
    plt.xlabel('distance to ground truth')
    plt.ylabel('reward')
    plt.show()
    plt.figure()
    plt.scatter(point[:,0],point[:,1])
    plt.show()
    return bins

def debug_plot(bins_generated_traj,bins_real_traj):
    plt.figure()
    plt.plot(bins_generated_traj,'go',label = 'generated_traj')
    plt.plot(bins_real_traj,'ro',label = 'real_traj')
    plt.xlabel('distance to ground truth')
    plt.ylabel('reward')
    plt.legend(loc = 'best')
    plt.title('comparison on reward between real traj. and generated traj.(test set)')
    plt.show()
    
bin_num = 100
bins_real_traj = dist_reward_plot_for_matrix(Dis_x,bin_num,s_a_cross_reward_all)


a_list_test = []
for tmp in hsc_traj_list:
    if np.random.rand() > 0.9:
        a_list_test.append(tmp)

s_a_cross_reward_all_policy = s_a_cross_test_multi_traj_multi_step(lmsdata_test,x_ego_test,lmsdata_diff_test,lmsdata_pre_test,a_list_test,global_save_dir,T_start_s = 48,T_start_a = 0, M_test = 20)
plt.figure()
plt.imshow(s_a_cross_reward_all_policy,cmap = 'Greys_r')
plt.colorbar()
plt.show()
Dis_x_hsc = L2_dis_matrix(a_list_test,0,29,x_ego_test,48,77)
    
bins_generated_traj = dist_reward_plot_for_matrix(Dis_x_hsc,bin_num,s_a_cross_reward_all_policy)

debug_plot(bins_generated_traj,bins_real_traj)

arg_max_s_a_cross_reward_all = np.argmax(s_a_cross_reward_all,axis = 0)
arg_max_s_a_cross_reward_all_policy = np.argmax(s_a_cross_reward_all_policy,axis = 0)

def selection_plot(arg_max_s_a_cross_reward_all,Dis):
    Dis_x_selected = []
    for i in np.arange(np.size(arg_max_s_a_cross_reward_all) ):
        Dis_x_selected.append( Dis[i ,arg_max_s_a_cross_reward_all[i]] )
    #hist,edge = np.histogram(Dis_x_selected, bin_num, (0,max(Dis_x_selected)+0.001) )
    
    n,edge,_ = plt.hist(Dis_x_selected, bin_num, (0,max(Dis_x_selected)+0.001),normed=True, label = 'selected by reward' )
    _,_,_ = plt.hist(np.reshape(Dis,[-1]),bin_num,(0,max(Dis_x_selected)+0.001),normed=True,label = 'random choices')
    plt.xlabel('distance to ground truth')
    plt.ylabel('frequency')
    plt.legend(loc='best')
    plt.title('precision_fig')
    plt.show()

selection_plot(arg_max_s_a_cross_reward_all,Dis_x)
selection_plot(arg_max_s_a_cross_reward_all,Dis_x_hsc)
    
"""log:
cur_nav = nav_list[11]#left lane change
cur_traj = traj_list[12]#right lane change
cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
nav_test = x_ego1    
test_one_traj_multi_step(lmsdata1[48:],lmsdata1_diff[48:],a1[48:],x_ego1[48:],global_save_dir,50)
-1.53563654
test_one_traj_multi_step(lmsdata[1][48:],lmsdata_diff[1][48:],a_list[0][48:],x_ego[1][48:],global_save_dir,50)
-0.43818265
a1 = a_list[0]
lmsdata1[48] almost equal to lmsdata[1][48]


reason: x_ego contain speed (which is different), and the laser is also slightly different

"""


"""
# important demo"""

cur_nav = nav_list[0]#left lane change
cur_traj = traj_list[10]#right lane change
s_a_cross_test(yaw_test_range,vdiff_test_range,cur_nav,cur_traj,dt,depict = 'sa_mismatch')

cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
nav_test = x_ego1
"""
total_reward = test_one_traj_multi_step(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir,80)
print 'mismatch',total_reward
test_one_traj(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir)
"""

"""

cur_nav = nav_list[10]#left lane change
cur_traj = traj_list[10]#right lane change
s_a_cross_test(yaw_test_range,vdiff_test_range,cur_nav,cur_traj,dt,depict = 'sa_match')

cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata2_pre = process_Data_real(cur_nav,cur_traj,dt)
nav_gt = x_ego1
total_reward = test_one_traj_multi_step(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir,80)
print 'match',total_reward
test_one_traj(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir)
"""


def state_gds_one_traj(s_lms,s_lmsdiff,a,s_ego,save_dir,M_test = 100):
    """
    return:(T,num_of_lms_lines)
    """
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    ind_hash = np.arange(1,dtype = np.int32)
    
    with tf.Graph().as_default():#creat a new clean graph
            solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                       start_point_range = 1,  # selected the start point of training sequence should < this   
                       update_rule='sgd',
                       memory_frame = M_test,
        #               hidden_size = 100,
                       batch_size = 1,#can remove(removed: duplicated calulation but no bugs)
                       num_epochs=0,
                       optim_config={
                         'learning_rate': 3e-5,#0:creat a network with ini_state
                       },
                       depict = 'irl',
                       test_only = True,
                       print_every = 1000,
                       save_dir = global_save_dir,
                       sampling_range_list = np.array([[0,0],[1.,8.]]),
                       num_of_lms_lines = num_of_lms_lines,
                       state_decode_only = True
                       )
            solver_test.state_single_step_gradient(s_norm_new, a_norm_new)
    gds = solver_test.state_decode_gds_list
    gds = gds[0][0][0][:,1,:]
    state_gds = gds[:,num_of_lms_lines:2*num_of_lms_lines]
    
    return state_gds

def state_decode_one_traj(s_lms,s_lmsdiff,a,s_ego,save_dir,M_test = 100):
    """
    developing
    for sa_norm_pointer
    """
    s_norm_new,a_norm_new = sa_norm_pointer(s_lms,s_lmsdiff,s_ego,a)
    ind_hash = np.arange(1,dtype = np.int32)
    
    with tf.Graph().as_default():#creat a new clean graph
            solver_test   = IRL_Solver_demo21(s_norm_new, a_norm_new, ind_hash,
                       start_point_range = 1,  # selected the start point of training sequence should < this   
                       update_rule='sgd',
                       memory_frame = M_test,
                       num_of_lms_lines = num_of_lms_lines,
        #               hidden_size = 100,
                       batch_size = 1,#can remove(removed: duplicated calulation but no bugs)
                       num_epochs=0,
                       optim_config={
                         'learning_rate': 3e-5,#0:creat a network with ini_state
                       },
                       depict = 'irl',
                       test_only = True,
                       print_every = 1000,
                       save_dir = global_save_dir,
                       sampling_range_list = np.array([[0,0],[1.,8.]]),
                       state_decode_only = True
                       )
            solver_test.state_decode(learning_rate = 1.,max_iteration = 100)
            plt.plot(solver_test.state_decode_loss_listory)
            val = solver_test.s_decode_val[0]
    
    lmsdata_decode = 1/val[0,:,num_of_lms_lines:num_of_lms_lines*2]
    return lmsdata_decode

"""gds_check
T = 150
cur_nav = nav_list[10]#left lane change
cur_traj = traj_list[10]#right lane change

cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
#lmsdata1_decode = state_decode_one_traj(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir,80)
single_step_gds_decode = state_gds_one_traj(lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir,80)
traj_animation(x_ego1,lmsdata1,single_step_gds_decode)
"""
    
"""
cur_nav = nav_list[0]#left lane change
cur_traj = traj_list[10]#right lane change
cur_nav = cur_nav[:T]
cur_traj = cur_traj[:T]
x_ego1,lmsdata1,lmsdata1_diff,a1,lmsdata1_pre = process_Data_real(cur_nav,cur_traj,dt)
plot_nav_at_T_compair(nav_test,nav_gt,lmsdata1[40],40)
plot_navs(nav_test,lmsdata1[40],x_ego,T = 40,x_ratio = 10,y_ratio = 2)
"""
