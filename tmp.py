
#      if self.test_only == False and self.load_train == False:
#          init = tf.initialize_all_variables()
#          sess.run(init)
#      else:
#          tf.train.Saver().restore(sess, self.save_dir+'params')
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import reward_v02
from irl_solver_tf import *
from scipy.interpolate import interp1d


s_lms,s_lmsdiff,a,s_ego,save_dir,M_test = lmsdata1,lmsdata1_diff,a1,x_ego1,global_save_dir,80

global_save_dir = './save/demo16/'

s_norm_new,a_norm_new = sa_norm_v03(s_lms,s_lmsdiff,s_ego,a)
ind_hash = np.arange(1,dtype = np.int32)

with tf.Graph().as_default():#creat a new clean graph
        solver_test   = IRL_Solver_demo16(s_norm_new, a_norm_new, ind_hash,
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
                   state_decode_only = True
                   )
        solver_test.state_single_step_gradient(s_norm_new, a_norm_new)
gds = solver_test.state_decode_gds_list
gds = gds[0][0][0][:,1,:]
state_gds = gds[:,num_of_lms_lines:2*num_of_lms_lines]