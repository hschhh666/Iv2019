#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:52:52 2018

@author: sry
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import tensorflow as tf
#from rnn_cell_dev import *
#from q2_initialization import xavier_weight_init


def block_lms_point_by_line(lms_frame,ang_frame,p1,p2,res):
    """
    p1:(2) [x,y]
    ps:(2) [x,y]
    lms_frame:(num_of_lms_point + ?,)
    res:(1,) resolution
    ang_frame:(num_of_lms_point,) increasing sequence, ang of each lms_point
    """

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
        lw1 = np.array([-100,init_s['lw']]) - x_ego_frame[:2]
        lw2 = np.array([1100,init_s['lw']])- x_ego_frame[:2]
        rw1 = np.array([-100,init_s['rw']])- x_ego_frame[:2]
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

def update(num, navdata, line, line_forward, lms_data = None, lms_line = None, predict_data = None, predict_line = None, predict_lines = None):
    """
    Feed(update) data to the ploted vectors
    """
    ax.set_xlim(navdata[num,0]-20.0,navdata[num,0]+20.0)
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




        
T = 100
tmpv = np.linspace(10.,10.,T)
tmpyaw = np.sin(np.linspace(0.,np.pi,T))*0.15
#tmpyaw = np.linspace(np.pi/2.,np.pi/2.,100)
x_ego = v2x(tmpv,tmpyaw,0.1,(0,0))
env_cars = np.tile(np.array([[[25,6,0],[50,1,0],[30,7,0.2],[19,9,-0.1],[59,4,0.06]]]),[T,1,1])
num_of_lms_lines = 361
init_s = {}
init_s['lw'] = 12.0 
init_s['rw'] = -2.0
lmsdata = x2lms(init_s,x_ego,env_cars,num_of_lms_lines)


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
line_ani = animation.FuncAnimation(fig1, update, T, fargs=(x_ego, l, t, lmsdata, j, None, k, g),
                                   interval=50, blit=False)
plt.show()




    
    



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