#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:51:43 2018

@author: sry
"""
import numpy as np



def creat_index_bias_matrix2(step_size,memory_frame,window_size_half,label_step_delay):
    """
    example:
        creat_index_bias_matrix2(2,5,3,4)
     
        array([[-2,  0,  2,  4 = label_step_delay,  6,  8, 10],
               [ 0,  2,  4,  6                   ,  8, 10, 12],
               [ 2,  4,  6,  8                   , 10, 12, 14],
               [ 4,  6,  8, 10                   , 12, 14, 16],
               [ 6,  8, 10, 12                   , 14, 16, 18]])
    """
    
    #add 'window' dim
    tmp2 = np.arange( -window_size_half*step_size + label_step_delay, window_size_half*step_size+1 + label_step_delay, step_size)
    tmp3 = tmp2 + np.expand_dims(np.arange(0,memory_frame*step_size,step_size),1)
    
    return tmp3
    
    
def creat_index_bias_matrix3(step_size,memory_frame,window_size):
    """
    example:
        creat_index_bias_matrix3(2,6,5)

        array([[ 0,  2,  4,  6,  8],
               [ 2,  4,  6,  8, 10],
               [ 4,  6,  8, 10, 12],
               [ 6,  8, 10, 12, 14],
               [ 8, 10, 12, 14, 16],
               [10, 12, 14, 16, 18]])
    """
    
    #add 'window' dim
    tmp2 = np.arange( 0, (window_size - 1)*step_size + 1 , step_size)
    tmp3 = tmp2 + np.expand_dims(np.arange(0,memory_frame*step_size,step_size),1)
    
    return tmp3
    
def creat_batch_window_index(tmp3,start_line):
    """
    Input:
    - start_line: (N,); N is batch_size
    - tmp3: create by function 'creat_index_bias_matrix2'
    
    example:
        
        creat_batch_window_index(creat_index_bias_matrix2(2,5,3,4),[1,11,21])
        =>
        array([[[-1,  1,  3,  5,  7,  9, 11],
                [ 1,  3,  5,  7,  9, 11, 13],
                [ 3,  5,  7,  9, 11, 13, 15],
                [ 5,  7,  9, 11, 13, 15, 17],
                [ 7,  9, 11, 13, 15, 17, 19]],
        
               [[ 9, 11, 13, 15, 17, 19, 21],
                [11, 13, 15, 17, 19, 21, 23],
                [13, 15, 17, 19, 21, 23, 25],
                [15, 17, 19, 21, 23, 25, 27],
                [17, 19, 21, 23, 25, 27, 29]],
        
               [[19, 21, 23, 25, 27, 29, 31],
                [21, 23, 25, 27, 29, 31, 33],
                [23, 25, 27, 29, 31, 33, 35],
                [25, 27, 29, 31, 33, 35, 37],
                [27, 29, 31, 33, 35, 37, 39]]])
        
        creat_batch_window_index(creat_index_bias_matrix3(2,5,3),[1,11,21])
        =>
        array([[[ 1,  3,  5],
                [ 3,  5,  7],
                [ 5,  7,  9],
                [ 7,  9, 11],
                [ 9, 11, 13]],
        
               [[11, 13, 15],
                [13, 15, 17],
                [15, 17, 19],
                [17, 19, 21],
                [19, 21, 23]],
        
               [[21, 23, 25],
                [23, 25, 27],
                [25, 27, 29],
                [27, 29, 31],
                [29, 31, 33]]])
    """
    return tmp3 + np.expand_dims(np.expand_dims(start_line,1),1)

