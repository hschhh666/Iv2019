#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:41:38 2018

@author: sry
"""
import tensorflow as tf
import numpy as np
from rnn_cell_dev import _linear
from q2_initialization import xavier_weight_init



    
def ANN_layer(x_input,hidden_dim,Dout,):
    """
    tensorflow implements
    """
    N, Din = x_input.get_shape()
    with tf.variable_scope("ANN_layer"):
        W1 = tf.get_variable('W1',(Din, hidden_dim), initializer = xavier_weight_init(),dtype = self.dtype)
        b1 = tf.get_variable('b1',hidden_dim, initializer=tf.constant_initializer(0.0),dtype = self.dtype)
        W2 = tf.get_variable('W2',(hidden_dim, Dout), initializer = xavier_weight_init(),dtype = self.dtype)
        b2 = tf.get_variable('b2',Dout, initializer=tf.constant_initializer(0.0),dtype = self.dtype)
        proj1 = tf.matmul(x_input, W1) + b1
        output = tf.matmul(tf.tanh(proj1), W2) + b2
        return output

def lstm_layer(x_input,hidden_dim = -1,states_prev = (None,None)):
    """
    states_prev: tuple of 2-D tensor (c_prev, h_prev), can use (none, none) as default
    hidden_dim or states_prev must be given at least one!
    inputs: list of tensor
    """
    if type(x_input) != list:
       N, T, _ = x_input.get_shape()
       #inputs = [ tf.squeeze(x, [1]) for x in tf.split(1, T , x_input)]#(N,T,D) -> T x (N,D)
       inputs = [ tf.squeeze(x, [1]) for x in tf.split(x_input, T , 1)]#(N,T,D) -> T x (N,D)
    else:
       N,_ = x_input[0].get_shape()
       inputs = x_input

    rnnOutput = []
    rnn = BasicLSTMCell(hidden_dim,state_is_tuple = True)
    c_prev, h_prev = rnn.zero_state(N)
    if(states_prev[0] == None):
        states_prev[0] = c_prev
    else:
        hidden_dim = states_prev[0].get_shape()[1]
    if(states_prev[1] == None):
        states_prev[1] = h_prev
    else:
        hidden_dim = states_prev[1].get_shape()[1]
    if hidden_dim == -1:
       raise ValueError("hidden_dim or states_prev must be given at least one!")
   
    with tf.variable_scope("lstm_layer"):
      for step, a_rnn_input in enumerate(inputs):#D: lead_dis speed acc ego_v rel_v
        if step > 0:
          tf.get_variable_scope().reuse_variables()
        one_rnnOutput, rnnstates_prev,_ = rnn(a_rnn_input,states_prev,verbose = False)
        rnnOutput.append(one_rnnOutput)#T x (N,h_size)
    return rnnOutput

def projection_layer(x_input,Dout):
    with tf.variable_scope('proj_layer'):
        N, Din = x_input.get_shape()
        W1 = tf.get_variable('W1',(Din, Dout), initializer = xavier_weight_init(),dtype = self.dtype)
        b1 = tf.get_variable('b1',Dout, initializer=tf.constant_initializer(0.0),dtype = self.dtype)  
        proj1 = tf.matmul(x_input, W1) + b1
        return proj1

def loop_conv1d(value, kernel_width, channels, stride, use_cudnn_on_gpu=True, name=None):
    """
    value: (N,D,C)
    """
    N, D, C = value.get_shape()
    
    padwidth = np.int(np.floor(stride/2))
    """
        there's a bug when if padwidth == 0
            padwidth will be 0,
            then, "[padwidth:-padwidth]" will be nothing,
            and [0:] will be the same with [-0:]
    """
    padwidth = np.max([padwidth,1])
    
    with tf.variable_scope('conv1d'):
        
        filters = tf.get_variable('kernel',(kernel_width, C.value, channels),initializer = xavier_weight_init())
        bias = tf.get_variable('bias',channels,initializer = tf.constant_initializer(0))
        
        value = tf.concat(1,[ value[:,-padwidth:,:], value, value[:,:padwidth,:] ] )
        conv = tf.nn.conv1d(value, filters, stride,'SAME', use_cudnn_on_gpu=use_cudnn_on_gpu, data_format="NHWC", name=name)
        add_bias = tf.nn.bias_add(conv, bias, data_format="NHWC")
        output = tf.nn.relu(add_bias)#N,D,C
        output = output[:,padwidth:-padwidth,:]
    return output
    

    
    
def reward_v101(s,a,params, trainable = True):
   """
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
       #a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   hidden_state = None
   Rewards = []
   s_prediction_list = []
   Diff = []
   with tf.variable_scope("reward_v01"):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           s_prediction_list.append(s_cur)
           if step > 0:
               tf.get_variable_scope().reuse_variables()
           s_encode = _linear(s_cur,s_encode_dim,True,scope = 'Proj', trainable = trainable)
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable))
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable) )
           one_reward = _linear(hidden_state,1,True,scope = 'Wout', trainable = trainable)
           s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable)
           s_prediction_list.append(s_encode_prediction)
           Rewards.append(one_reward)


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d,s_prediction_list



def reward_v1_3(s,a,params, trainable = True):
   """
   residual structure + state_diff structure
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   
   def parse_scur(s_cur,num_of_lms_lines):
       s_lms = s_cur[:,:num_of_lms_lines]
       s_lmsdiff = s_cur[:,num_of_lms_lines:2*num_of_lms_lines]
       s_ego = s_cur[:,2*num_of_lms_lines:]
       return s_lms,s_lmsdiff,s_ego
   
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
    #    a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a,T,1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   num_of_lms_lines = params['num_of_lms_lines']
   hidden_state = None
   Rewards = []
   s_prediction_list = []
   Diff = []
   with tf.variable_scope("reward_v01",regularizer = tf.contrib.layers.l2_regularizer(params['regularizer_scale'])):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           s_prediction_list.append(s_cur)
           if step > 0:
               tf.get_variable_scope().reuse_variables()
              
#           #conv0 part
#           s_lms,s_lmsdiff,s_ego = parse_scur(s_cur,num_of_lms_lines)
#           s_lms,s_lmsdiff = tf.expand_dims(s_lms,2),tf.expand_dims(s_lmsdiff,2)
#           conv0 = tf.concat(2,[s_lms,s_lmsdiff])
#           
#           #conv part (value, kernel_width, channels, stride, use_cudnn_on_gpu=True, name=None)
#           with tf.variable_scope("conv1"):
#               conv1 = loop_conv1d(conv0,3,channels = 4, stride = 2)
#           with tf.variable_scope("conv2"):
#               conv2 = loop_conv1d(conv1,3,channels = 8, stride = 2)
#           with tf.variable_scope("conv3"):
#               conv3 = loop_conv1d(conv2,3,channels = 16, stride = 2)
#           
#           # 'bounding box regresion ,1x1conv'
#           with tf.variable_scope("vehicle_position_regresion"):
#               conv4 = loop_conv1d(conv3,1,channels = 1, stride = 1)
#           conv4_squeeze = tf.squeeze(conv4,squeeze_dims = [2])

               
           # linear
#           s_encode = _linear([conv4_squeeze,s_ego],s_encode_dim,True,scope = 's_encode', trainable = trainable)
           s_encode = _linear(s_cur,s_encode_dim,True,scope = 's_encode', trainable = trainable)
           
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable))
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
               tmp = s_encode_prediction
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + tmp
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable) )
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + s_encode
           one_reward = _linear(hidden_state,1,True,scope = 'Wout', trainable = trainable)
           s_prediction_list.append(s_encode_prediction)
           Rewards.append(one_reward)
#       print conv1.get_shape()
#       print conv2.get_shape()
#       print conv3.get_shape()
#       print conv4.get_shape()


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d,s_prediction_list
   
   
def reward_v1_2(s,a,params, trainable = True):
   """
   residual structure + state_diff structure
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   
   def parse_scur(s_cur,num_of_lms_lines):
       s_lms = s_cur[:,:num_of_lms_lines]
       s_lmsdiff = s_cur[:,num_of_lms_lines:2*num_of_lms_lines]
       s_ego = s_cur[:,2*num_of_lms_lines:]
       return s_lms,s_lmsdiff,s_ego
   
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
       #a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   num_of_lms_lines = params['num_of_lms_lines']
   hidden_state = None
   Rewards = []
   s_prediction_list = []
   Diff = []
   with tf.variable_scope("reward_v01"):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           s_prediction_list.append(s_cur)
           if step > 0:
               tf.get_variable_scope().reuse_variables()
              
           #conv0 part
           s_lms,s_lmsdiff,s_ego = parse_scur(s_cur,num_of_lms_lines)
           s_lms,s_lmsdiff = tf.expand_dims(s_lms,2),tf.expand_dims(s_lmsdiff,2)
           conv0 = tf.concat(2,[s_lms,s_lmsdiff])
           
           #conv part (value, kernel_width, channels, stride, use_cudnn_on_gpu=True, name=None)
           with tf.variable_scope("conv1"):
               conv1 = loop_conv1d(conv0,3,channels = 4, stride = 2)
           with tf.variable_scope("conv2"):
               conv2 = loop_conv1d(conv1,3,channels = 8, stride = 2)
           with tf.variable_scope("conv3"):
               conv3 = loop_conv1d(conv2,3,channels = 16, stride = 2)
           
           # 'bounding box regresion ,1x1conv'
           with tf.variable_scope("vehicle_position_regresion"):
               conv4 = loop_conv1d(conv3,1,channels = 1, stride = 1)
           conv4_squeeze = tf.squeeze(conv4,squeeze_dims = [2])

               
           # linear
           s_encode = _linear([conv4_squeeze,s_ego],s_encode_dim,True,scope = 's_encode', trainable = trainable)
#           s_encode = _linear(s_cur,s_encode_dim,True,scope = 's_encode', trainable = trainable)
           
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable))
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
               tmp = s_encode_prediction
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + tmp
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable) )
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + s_encode
           one_reward = _linear(hidden_state,1,True,scope = 'Wout', trainable = trainable)
           s_prediction_list.append(s_encode_prediction)
           Rewards.append(one_reward)
       print ( conv1.get_shape() )
       print ( conv2.get_shape() )
       print ( conv3.get_shape() )
       print ( conv4.get_shape() )


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d,s_prediction_list
   
def reward_v1_1(s,a,params, trainable = True):
   """
   residual structure
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
       #a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   hidden_state = None
   Rewards = []
   s_prediction_list = []
   Diff = []
   with tf.variable_scope("reward_v01"):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           s_prediction_list.append(s_cur)
           if step > 0:
               tf.get_variable_scope().reuse_variables()
           s_encode = _linear(s_cur,s_encode_dim,True,scope = 'Proj', trainable = trainable)
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable))
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
               tmp = s_encode_prediction
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + tmp
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable) )
               s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable) + s_encode
           one_reward = _linear(hidden_state,1,True,scope = 'Wout', trainable = trainable)
           s_prediction_list.append(s_encode_prediction)
           Rewards.append(one_reward)


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d,s_prediction_list
    
def reward_v1(s,a,params, trainable = True):
   """
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
       #a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   hidden_state = None
   Rewards = []
   s_prediction_list = []
   Diff = []
   with tf.variable_scope("reward_v01"):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           s_prediction_list.append(s_cur)
           if step > 0:
               tf.get_variable_scope().reuse_variables()
           s_encode = _linear(s_cur,s_encode_dim,True,scope = 'Proj', trainable = trainable)
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable))
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win', trainable = trainable) )
           one_reward = _linear(hidden_state,1,True,scope = 'Wout', trainable = trainable)
           s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH', trainable = trainable)
           s_prediction_list.append(s_encode_prediction)
           Rewards.append(one_reward)


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d,s_prediction_list
   
   
def reward_v01(s,a,params):
   """
   s:current state descriptor,(N,T,Dh)
   a:action sequence, list or sequence, of (N,T,Din), action list
   params:index
       - hidden_dim
       - s_encode_dim
       
   return: (N,1),(N,1)
   """
   if type(a) != list:
       N, T, _ = a.get_shape()
       T_val = T.value
       #a = [ tf.squeeze(x, [1]) for x in tf.split(1, T , a)]#(N,T,D) -> T x (N,D)
       a = [ tf.squeeze(x, [1]) for x in tf.split(a, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = a[0].get_shape()
       T_val = len(a)

   if type(s) != list:
       N, T, _ = s.get_shape()
       T_val = T.value
       #s = [ tf.squeeze(x, [1]) for x in tf.split(1, T , s)]#(N,T,D) -> T x (N,D)
       s = [ tf.squeeze(x, [1]) for x in tf.split(s, T , 1)]#(N,T,D) -> T x (N,D)
   else:
       N,_ = s[0].get_shape()
       T_val = len(s)
       
   hidden_dim = params['hidden_dim']
   s_encode_dim = params['s_encode_dim']
   hidden_state = None
   Rewards = []
   Diff = []
   with tf.variable_scope("reward_v01"):
       for step, a_cur in enumerate(a):
           s_cur = s[step]
           if step > 0:
               tf.get_variable_scope().reuse_variables()
           s_encode = _linear(s_cur,s_encode_dim,True,scope = 'Proj')
           if step > 0:
               hidden_state =  tf.tanh( _linear([s_encode_prediction,a_cur],hidden_dim,True,scope = 'Win') )
               one_Diff = tf.reduce_mean(tf.square(s_encode_prediction-s_encode),reduction_indices=1,keep_dims = True)
               Diff.append(one_Diff)
           else:
               hidden_state =  tf.tanh( _linear([s_encode,a_cur],hidden_dim,True,scope = 'Win') )
           one_reward = _linear(hidden_state,1,True,scope = 'Wout')
           s_encode_prediction = _linear(hidden_state,s_encode_dim,True,scope = 'WH')   
           Rewards.append(one_reward)


   r = tf.add_n(Rewards) / T_val

   d = tf.zeros_like(r)
   if T_val>1:
       d = tf.add_n(Diff) / (T_val-1)
       
   return r,d
           
           
           
       
       
#   hidden_dim = params['hidden_dim']
#   
#   states_prev = (None,projection_layer(s,hidden_dim))
#
#   Outputs = []
#   rnn = BasicLSTMCell(hidden_dim,state_is_tuple = True)
#   c_prev, h_prev = rnn.zero_state(N)
#   if(states_prev[0] == None):
#        states_prev[0] = c_prev
#   else:
#        hidden_dim = states_prev[0].get_shape()[1]
#   if(states_prev[1] == None):
#        states_prev[1] = h_prev
#   else:
#        hidden_dim = states_prev[1].get_shape()[1]
#   
#   with tf.variable_scope("reward_v01"):
#      for step, a_rnn_input in enumerate(a):#D: lead_dis speed acc ego_v rel_v
#        if step > 0:
#          tf.get_variable_scope().reuse_variables()
#        one_rnnOutput, rnnstates_prev,_ = rnn(a_rnn_input,states_prev,verbose = False)
#        
#        one_Output = _linear(one_rnnOutput,1,True)
#        Outputs.append(one_Output)#T x (N,h_size)  
#        
#        
#        
#   r = tf.add_n(Outputs) / T_val
#   return r

def reward_v02(s,a,params):
   """
   pure single-step!
   s:current state descriptor,(N,Dh)
   a:(N,Din), one action
   params:index
       - hidden_dim
   """
   
   hidden_dim = params['hidden_dim']
   states_prev =  tf.tanh( _linear([s,a],hidden_dim,True,scope = 'Win') )
   one_Output = _linear(states_prev,1,True,scope = 'Wout')
   r = one_Output
   return r

    


        
        
        
class demo_model(object):
    """
    demo, using Multi-layer perceptron, take lms as input, generate control sequence
    tensorflow implement
    """
    def _init_(self):
        return 0
    
    