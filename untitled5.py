#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:48:17 2018

@author: sry

use real data (traj from highwayMODT), trainset overfitting experiments
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import reward_v02
from irl_solver_tf import *


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




def plot_lms_point(lms_frame,ang_frame):
    return 0



def creat_lms_frame(init_s,x_ego_frame,env_cars,num_of_lms_lines):
    """
    init_s['lw'] :(1,) left wall position
    init_s['rw'] :(1,) right wall position
    x_ego_frame:(3,) x,y,yaw
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
    x,y,yaw = x_ego_frame
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
    for i in xrange(n):
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
    if init_s != None:
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
    for i in xrange(T):
        lms[i,:] = creat_lms_frame(init_s,x_ego[i,:],env_cars[i,:,:],num_of_lms_lines)[:num_of_lms_lines]
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
    

def update(num, navdata, line, line_forward, lms_data = None, lms_line = None, predict_data = None, predict_line = None, predict_lines = None):
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
    if lms_data != None:
        _,H = lms_data.shape
        lmsFrame = lms_data[num,:]
        if (H/2)*2 != H:
            lmsYaw = navdata[num,2] + np.arange(-H/2+1,H/2+1) *2.0 * np.pi/(H-1)
        else:
            lmsYaw = navdata[num,2] + (np.arange(-H/2,H/2)+0.5 )*2.0 * np.pi/H
        lmsX = lmsFrame * np.cos(lmsYaw)  + navdata[num,0]
        lmsY = lmsFrame * np.sin(lmsYaw)  + navdata[num,1]
        lms_line.set_data( np.vstack((lmsX,lmsY)) )


        
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
#    plt.plot(np.vstack((lmsX,lmsY)),'g*') 

def visulization_0(nav,lmsFrame,reward_test):
    plot_nav(nav,lmsFrame)
    plt.figure()
    plt.imshow( np.flipud(reward_test.T),cmap = 'Greys_r' )
    plt.xlabel('T')
    plt.ylabel('steering angle(idx)')
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
    
dir_path = '/media/sry/DATA/bp/new_story/ex/lane_pose_0/'




nav_list = []
traj_list = []

#bug data:212

#np.arange(1,414)
for i in np.arange(211,222):
    navfile = dir_path + 'nav%d'%i
    trajfile = dir_path + 'traj%d'%i
    nav = sio.loadmat(navfile)
    nav = nav['nav_list']
    nav_list.append(nav)
    traj = sio.loadmat(trajfile)
    traj = traj['new_traj']
    traj_list.append(traj)

init_s = None
#init_s = {}
#init_s['lw'] = 12.0 
#init_s['rw'] = -2.0
dt = 0.1
num_of_lms_lines = 361

def process_Data_real(cur_nav,cur_traj,dt):
    cur_traj = np.transpose(cur_traj,axes = [1,0,2])
    #traj:milli, fno, gp.x, gp.y, glen0, glen1, gv1.x, gv1.y, ep.x, ep.y, ev1.x, ev1.y, interfrmspd
    tmpv,tmpyaw = x2yaw(cur_nav[:,1],cur_nav[:,2],dt)
    tmpyawdif = np.append( np.diff(tmpyaw),0)/dt
    tmpvdif = np.append(np.diff(tmpv),0)/dt
    a = np.concatenate([np.array([tmpvdif]),np.array([tmpyawdif])],axis = 0).T
    x_ego1 = np.concatenate([cur_nav[:,[1]],cur_nav[:,[2]],np.expand_dims(tmpyaw,axis = 1)],axis = 1)
    env_cars = np.concatenate( (cur_traj[:,:,1:3],np.zeros_like(cur_traj[:,:,[1]]) ), axis = 2)
    lmsdata1 = x2lms(init_s,x_ego1,env_cars,num_of_lms_lines)
    lmsdata1_diff = np.concatenate([lmsdata1[1:,]-lmsdata1[:-1,], [np.zeros(num_of_lms_lines)]],axis = 0)
    return x_ego1,lmsdata1,lmsdata1_diff,a


cur_nav = nav_list[0]
cur_traj = traj_list[0]
x_ego1,lmsdata1,lmsdata1_diff,a1 = process_Data_real(cur_nav,cur_traj,dt)
    
cur_nav = nav_list[2]
cur_traj = traj_list[2]
x_ego2,lmsdata2,lmsdata2_diff,a2 = process_Data_real(cur_nav,cur_traj,dt)

#x_ego = np.array([x_ego1,x_ego2,x_ego3,x_ego4,x_ego5])
#lmsdata = np.array([lmsdata1,lmsdata2,lmsdata3,lmsdata4,lmsdata5])
#lmsdata_diff = np.array([lmsdata1_diff,lmsdata2_diff,lmsdata3_diff,lmsdata4_diff,lmsdata5_diff])
#a = np.array([a1,a2,a3,a4,a5])


#x_ego = np.array([x_ego1,x_ego2,x_ego3])
#lmsdata = np.array([lmsdata1,lmsdata2,lmsdata3])
#lmsdata_diff = np.array([lmsdata1_diff,lmsdata2_diff,lmsdata3_diff])
#a = np.array([a1,a2,a3])



x_ego = [x_ego1,x_ego2]
lmsdata = [lmsdata1,lmsdata2]
lmsdata_diff = [lmsdata1_diff,lmsdata2_diff]
a = [a1,a2]


fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)




l, = ax.plot([], [], 'ro')#nav
j, = ax.plot([], [], 'g*')#lms
k, = ax.plot([], [], 'r-')#predict_previous
g, = ax.plot([], [], 'r-*')#predict
t, = ax.plot([], [], 'g+')#nav_after


ax.grid()
#plt.xlim(0, 2000)
#plt.ylim(-1000, 1000)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update, 100, fargs=(x_ego2, l, t, lmsdata2, j, None, k, g),
                                   interval=50, blit=False)
plt.show()
    

"""
v0.0.0.0.0.0.0.1
"""

def reshape_for_single_step(data_list):
    return np.concatenate(data_list,axis = 0)
        


s_ego = reshape_for_single_step(x_ego)
s_lms = reshape_for_single_step(lmsdata)
s_lmsdiff = reshape_for_single_step(lmsdata_diff)
#s = np.concatenate([s_lms,s_ego],axis = 1)
a = reshape_for_single_step(a)
    

def sa_norm_v01(s,a):
    s_norm = 1/s
    #s_norm = ( s - 16 )/np.sqrt(78)
#    a_norm = a[:,:]/np.sqrt(0.001)
    a_norm = a[:,:]/np.sqrt([10.,0.0001])
    return s_norm,a_norm

def sa_norm_v02(s_lms,s_lmsdiff,a):
    s_norm1 = np.tanh(s_lmsdiff)
    s_norm2 = 1/s_lms
    s_norm = np.concatenate([s_norm1,s_norm2],axis = 1)
    #s_norm = ( s - 16 )/np.sqrt(78)
#    a_norm = a[:,:]/np.sqrt(0.001)
    a_norm = a[:,:]/np.sqrt([50.,0.0002])
    return s_norm,a_norm
    

    
#def sa_norm_v03(s_lms,s_lmsdiff,????,a):
#    s_norm1 = np.tanh(s_lmsdiff)
#    s_norm2 = 1/s_lms
#    s_norm = np.concatenate([s_norm1,s_norm2],axis = 1)
#    #s_norm = ( s - 16 )/np.sqrt(78)
##    a_norm = a[:,:]/np.sqrt(0.001)
#    a_norm = a[:,:]/np.sqrt([1.,0.001])
#    return s_norm,a_norm
#    
s_norm,a_norm = sa_norm_v02(s_lms,s_lmsdiff,a)


mode = 'train'
if mode != 'load':
    
    solver   = IRL_Solver_demo6(s_norm, a_norm,
               start_point_range = 1,  # selected the start point of training sequence should < this   
               update_rule='sgd',
               batch_size=s_norm.shape[0],
#               hidden_size = 100,
               iteration=2000,
               optim_config={
                 'learning_rate': 0.5,#0:creat a network with ini_state
                 'decay':1
               },
               print_every=10,
               depict = 'irl',
               test_only = False,
               save_dir = './save/demo9/',
               sample_expert_ratio = 3,
               sampling_range_list = np.array([[0,0],[1.,5.]]),
               )
    solver.train()


    plt.plot(solver.reward_diff_history,label = 'reward_diff')
    plt.plot(solver.reward_expert_history,label = 'reward_expert')
    plt.plot(solver.reward_policy_history,label = 'reward_policy')
    plt.legend(loc='best')
    plt.title('rward history')
    #plt.savefig('%s/train_loss.jpg' % ann_solver2.save_dir)
    plt.show()


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
#
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
                
                
yaw_test = np.linspace(-np.pi/50,np.pi/50,21)
vdiff_test = np.linspace(-2.*dt,2.*dt)
lyaw, = np.shape(yaw_test)   
lv, = np.shape(vdiff_test)

lms_test = []
lms_test_diff = []
a_test = []

def creat_testing_data_v01(lmsdata_test_list,lmsdata_test_diff_list,yaw_test,vdiff_test):
    lms_test = []
    lms_test_diff = []
    a_test = []
    for step,lmsdata_test in enumerate(lmsdata_test_list):
        lmsdata_test_diff = lmsdata_test_diff_list[step]
        for data in grid_search3d(lmsdata_test,yaw_test,vdiff_test):
            lms_test.append(data[0])
            a_test.append([data[2],data[1]])
        for data in grid_search3d(lmsdata_test_diff,yaw_test,vdiff_test):
            lms_test_diff.append(data[0])
    s_lms = np.array(lms_test)
    s_lmsdiff = np.array(lms_test_diff)
    a = np.array(a_test)
    return s_lms,s_lmsdiff,a
     
#s_lms,s_lmsdiff,a = creat_testing_data_v01([lmsdata4,lmsdata5],[lmsdata4_diff,lmsdata5_diff],yaw_test,vdiff_test)
#s_lms,s_lmsdiff,a = creat_testing_data_v01([lmsdata1,lmsdata2,lmsdata3,lmsdata4,lmsdata5],[lmsdata1_diff,lmsdata2_diff,lmsdata3_diff,lmsdata4_diff,lmsdata5_diff],yaw_test,vdiff_test)
s_lms,s_lmsdiff,a = creat_testing_data_v01([lmsdata1,lmsdata2],[lmsdata1_diff,lmsdata2_diff],yaw_test,vdiff_test)


s_norm_new,a_norm_new = sa_norm_v02(s_lms,s_lmsdiff,a)

#a_norm_gt = a_gt[:,[1]]/np.sqrt(0.001)

with tf.Graph().as_default():#creat a new clean graph
    solver_test   = IRL_Solver_demo6(s_norm_new, a_norm_new,
               start_point_range = 1,  # selected the start point of training sequence should < this   
               update_rule='sgd',
#               hidden_size = 100,
               num_epochs=0,
               optim_config={
                 'learning_rate': 3e-5,#0:creat a network with ini_state
               },
               depict = 'irl',
               test_only = True,
               print_every = 1000,
               save_dir = './save/demo9/',
               sampling_range_list = np.array([[0,0],[1.,5.]])
               )
    solver_test.train()
reward_test = np.array(solver_test.reward_expert_history)
reward_test = np.reshape(reward_test,(-1,lyaw,lv))
#reward_test1 = reward_test[:,10,:]
reward_test1 = reward_test[:,:,25]
arg_max_reward = np.argmax(reward_test1,axis = 1)
visulization_0(x_ego1,lmsdata1[0,:],reward_test1)
plt.plot( arg_max_reward,'o')

reward_test1 = reward_test[:,10,:]
arg_max_reward = np.argmax(reward_test1,axis = 1)
visulization_0(x_ego1,lmsdata1[0,:],reward_test1)
plt.plot( arg_max_reward,'o')
    
"""

#creat a map
col1 = np.linspace(1.,1.,1000)
col0 = np.linspace(0.,0.,1000)
map_tmp = np.tile(col0,[40,1]).T
map1 = np.concatenate((np.array([col1]).T,map_tmp,np.array([col1]).T),axis = 1)

#initialize
# in init. state, ego_vehicle is set as zero point, direction are set as zero.
init_s = {}
init_s['lw'] = 15.0 #distance to left wall
init_s['rw'] = 5.0

#creat traj. (20s)
t_total = 20
dt = 0.5
num_of_lms_lines = 360

#case 1.
#env cars traj. 
env_cars = []
v_env = np.array([0,5])
v_env = np.tile(v_env,[(int)(t_total/dt),1])
x0_env = (20,0)
x_env = v2x(v_env,dt,x0_env)#(x,y,yaw)
env_cars.append(x_env)

#ego cars traj
vtmp = np.array([0,5]) #(yaw,v)
v_ego = np.tile(vtmp,[(int)(t_total/dt),1])
x_ego = v2x(v_ego,dt,(0,0))

#case 2.


"""