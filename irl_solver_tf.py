
"""
Created on Tue Jan 10 17:22:47 2017

@author: sry
"""
import numpy as np
import tensorflow as tf 
from tensorflow.python.framework import ops
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity
import time
import os
from model import *
from utilities import *


class Save_dir(object):
    def __init__(self,save_dir = './'):
        self.save_dir = save_dir
        return None

def creating_index_matrix(start_line,memory_frame,ind):
    """
    start_line:(N,)
    ind:(L,)
    """        
    N, = np.shape(start_line)
    L, = np.shape(ind)
    start_line = ind[start_line]
    batch_index = np.expand_dims(start_line,1)+np.arange(memory_frame)
    return batch_index
    
    
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
    
    example:
        start_line = [1,3,5]
        memory_frame = 4
        
        ==>:array([[1, 2, 3, 4],
                   [3, 4, 5, 6],
                   [5, 6, 7, 8]])
    
    
    """        
    N, = np.shape(start_line)
    batch_index = np.expand_dims(start_line,1)+np.arange(memory_frame)
    return batch_index

    

class IRL_Solver_demo21(object):
  """
  residual structure
  """
    
  def __init__(self, s, a, ind_hash, a_policy = None, a_policy_hash = None, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    
    
    
    self.dipict = 'v02'
    self.model = object

    
    self.a_policy_hash = a_policy_hash
    self.a_policy = a_policy
    
    
#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.num_of_lms_lines = kwargs.pop('num_of_lms_lines')
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 10000)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    self.state_decode_only = kwargs.pop('state_decode_only', False)
    self.regularizer_scale = kwargs.pop('regularizer_scale', 0.0)
    self.a_hash = kwargs.pop('a_hash', ind_hash) # for s_a_cross_test
    if self.test_only == True:
        self.iteration = np.int(np.ceil(ind_hash.shape[0]/np.float(self.batch_size)))
#    if self.test_only == True:
#        self.iteration = self.iteration = ind_hash.shape[0]/self.batch_size+1

    

    self.model_params = {}
    self.model_params['hidden_dim'] = self.hidden_size
    self.model_params['s_encode_dim'] = self.hidden_size
    self.model_params['num_of_lms_lines'] = self.num_of_lms_lines
    self.model_params['regularizer_scale'] = self.regularizer_scale
    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

    if self.state_decode_only == False:
        
        self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
        self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
        self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
    #    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    #    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
        self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
        
        
        with tf.variable_scope('reward'):
            self.reward_expert_batch,self.s_Diff_batch,_ = reward_v1_3(self.s_placeholder,self.a_expert_placeholder,self.model_params)
            tf.get_variable_scope().reuse_variables()
            self.reward_policy_batch,_,_ = reward_v1_3(self.s_placeholder_for_policy,self.a_policy_placeholder,self.model_params)
            
    
            
            
            w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
            w = w/tf.reduce_sum(w)
            w = tf.stop_gradient(w)
           
#for debug:
#            w = 1.0/self.batch_size_for_policy
            
            self.reward_samples_debug = tf.reduce_mean(self.reward_policy_batch)
            self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
#            self.reward_policy = tf.reduce_mean(self.reward_policy_batch)
            self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
    #        self.reward_policy = tf.reduce_mean(reward_policy)
            self.reward_diff = self.reward_expert - self.reward_policy
            self.s_Diff = tf.reduce_mean(self.s_Diff_batch)
    
    
            
        if self.test_only == False:
          
          global_step = tf.Variable(0, trainable=False)
          learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=1,decay_rate=self.optim_config['decay'])
            
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer( learning_rate)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
    #     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
    
          """#see regularized variable
          weights_list = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          print 'weights_list:', weights_list
          """
          self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
          self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff + self.regularization_loss)
          self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)


            
        self.reward_expert_history = []
        self.reward_policy_history = []
        self.reward_diff_history = []
        self.s_Diff_history = []
        self.regularization_loss_history = []
        self.reward_samples_debug_history = []
        self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
        print (self.tmp3)
        
        localtime = time.asctime(time.localtime(time.time()))
        #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )
        

        
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.a_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))


    
  def _step(self, sess = None):
      

      """
      For Hu: use trajectory generator @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      """
      if self.a_policy_hash is not None and self.a_policy is not None:
          num_of_policy_trajectory = len(self.a_policy_hash)#the total policy_trajectory you generate
          indice_of_selection = np.random.choice(num_of_policy_trajectory,self.batch_size_for_policy,replace = False)#select some of them to feed in the batch      
          #you can modify 'self.batch_size_for_policy' in line 114
          policy_a = self.a_policy[creating_index_matrix(indice_of_selection, self.memory_frame, self.a_policy_hash)]
          samples_density = np.ones(self.batch_size_for_policy)
          """
          end
          """
          
          
          
      else:
          """
          kde_sampling 
          """
          samples = self.kde_generator.sample(self.batch_size_for_policy)
          policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
          samples_density = np.exp(self.kde_estimator.score_samples(samples))#this line take lot of time

      
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.a_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_density
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val,regularization_loss_val, _,reward_samples_debug_val = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.regularization_loss,self.train_op,self.reward_samples_debug], feed_dict = feed)
          self.regularization_loss_history.append(regularization_loss_val)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, reward_samples_debug_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff, self.reward_samples_debug], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      self.reward_samples_debug_history.append(reward_samples_debug_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
      
  def prepare_sess(self):
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      return sess
      
  def train(self):
      
      saver = tf.train.Saver()
      sess = self.prepare_sess()
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print( '(Iteration %d / %d) : ' % (t + 1, self.iteration))
              if self.test_only == False:
                  print( 'regularization_l2_loss (scale adjusted):',  np.sqrt(self.regularization_loss_history[-1])/2)
              print( 'reward_diff:', self.reward_diff_history[-1])
              print ('reward_samples_debug', self.reward_samples_debug_history[-1])
              print ('s_Diff_debug_for_L2_loss (scale adjusted):', np.sqrt(self.s_Diff_history[-1])/2 )#check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)
          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()


          
  def state_single_step_gradient(self,s,a):
      """
      input:
          - self.s (T,Ds), the tested state sequence
          - self.a (T,Da), the tested action sequence
          - self.ind
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s
      
      T,_ = np.shape(s)
      ind_hash = np.arange(T-1,dtype = np.int32)
      self.batch_size = T-1
      self.start_line = np.arange(self.batch_size)
      self.memory_frame = 2
      
      minibatch_s = s[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]
      s_init = minibatch_s
#      s_init = minibatch_s[:,:1,:]
#      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = a[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]

      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      self.a_tmp_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_tmp_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate = 0.)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)                           
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(1):
          
          feed = {self.a_tmp_placeholder: minibatch_a}
          Diff_val,s_gds_val = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
#          self.state_decode_gds_list = np.reshape(self.state_decode_gds_list,[-1])
          self.state_decode_loss_listory.append(Diff_val)
          self.state_decode_loss_listory = np.reshape(self.state_decode_loss_listory,[-1])
          
          if t % 100 == 0:
              print (Diff_val)
              
      sess.close()
              

                    
  def state_decode(self,learning_rate = 3e-3,max_iteration = 1000):
      """
      input:
          - self.s (T,Ds), but only the first data is used
          - self.a (T,Da), the tested action sequence
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s          
      
      
      if self.memory_frame < 2:
          print ('Please set memory frame larger than 1')
          return
      batch_size_old = self.batch_size
      ind_hash_old = self.ind_hash
      self.ind_hash = np.arange(1,dtype = np.int32)#currently
      self.batch_size = 1#currently
      self._reset()
      
      
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      s_init = minibatch_s[:,:1,:]
      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
                           
      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      
      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_expert_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.AdadeltaOptimizer(learning_rate)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)
      self.train_op_for_state_decode = cur_optmizer.apply_gradients(self.gds_for_state_decode)
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(max_iteration):
          feed = {self.a_expert_placeholder: minibatch_a}
          Diff_val,s_gds_val,_ = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode,self.train_op_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
          self.state_decode_loss_listory.append(Diff_val)
          if t % 100 == 0:
              print (Diff_val)
          
      self.s_decode_name = [i.name for i in tf.train.variables.trainable_variables()]
      self.s_decode_val =  [i.eval(session = sess) for i in tf.train.variables.trainable_variables()]
      sess.close()
      
      self.batch_size = batch_size_old
      self.ind_hash = ind_hash_old
    


class IRL_Solver_demo20(object):
  """
  residual structure
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.num_of_lms_lines = kwargs.pop('num_of_lms_lines')
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 10000)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    self.state_decode_only = kwargs.pop('state_decode_only', False)
    self.regularizer_scale = kwargs.pop('regularizer_scale', 0.0)
    self.a_hash = kwargs.pop('a_hash', ind_hash) # for s_a_cross_test
    if self.test_only == True:
        self.iteration = np.int(np.ceil(ind_hash.shape[0]/np.float(self.batch_size)))
#    if self.test_only == True:
#        self.iteration = self.iteration = ind_hash.shape[0]/self.batch_size+1

    

    self.model_params = {}
    self.model_params['hidden_dim'] = self.hidden_size
    self.model_params['s_encode_dim'] = self.hidden_size
    self.model_params['num_of_lms_lines'] = self.num_of_lms_lines
    self.model_params['regularizer_scale'] = self.regularizer_scale
    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

    if self.state_decode_only == False:
        
        self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
        self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
        self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
    #    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    #    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
        self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
        
        
        with tf.variable_scope('reward'):
            self.reward_expert_batch,self.s_Diff_batch,_ = reward_v1_3(self.s_placeholder,self.a_expert_placeholder,self.model_params)
            tf.get_variable_scope().reuse_variables()
            self.reward_policy_batch,_,_ = reward_v1_3(self.s_placeholder_for_policy,self.a_policy_placeholder,self.model_params)
            
    
            
            
            w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
            w = w/tf.reduce_sum(w)
            w = tf.stop_gradient(w)
            
            self.reward_samples_debug = tf.reduce_mean(self.reward_policy_batch)
            self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
#            self.reward_policy = tf.reduce_mean(self.reward_policy_batch)
            self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
    #        self.reward_policy = tf.reduce_mean(reward_policy)
            self.reward_diff = self.reward_expert - self.reward_policy
            self.s_Diff = tf.reduce_mean(self.s_Diff_batch)
    
    
            
        if self.test_only == False:
          
          global_step = tf.Variable(0, trainable=False)
          learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=1,decay_rate=self.optim_config['decay'])
            
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer( learning_rate)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
    #     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
    
          """#see regularized variable
          weights_list = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          print 'weights_list:', weights_list
          """
          self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
          self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff + self.regularization_loss)
          self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)
    
        
    
            
        self.reward_expert_history = []
        self.reward_policy_history = []
        self.reward_diff_history = []
        self.s_Diff_history = []
        self.regularization_loss_history = []
        self.reward_samples_debug_history = []
        self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
        print (self.tmp3)
        
        localtime = time.asctime(time.localtime(time.time()))
        #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )
        

        
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.a_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))


    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
#      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.a_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val,regularization_loss_val, _,reward_samples_debug_val = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.regularization_loss,self.train_op,self.reward_samples_debug], feed_dict = feed)
          self.regularization_loss_history.append(regularization_loss_val)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, reward_samples_debug_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff, self.reward_samples_debug], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      self.reward_samples_debug_history.append(reward_samples_debug_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
      
  def prepare_sess(self):
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      return sess
      
  def train(self):
      
      saver = tf.train.Saver()
      sess = self.prepare_sess()
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              if self.test_only == False:
                  print ('regularization_l2_loss (scale adjusted):',  np.sqrt(self.regularization_loss_history[-1])/2)
              print ('reward_diff:', self.reward_diff_history[-1])
              print ('reward_samples_debug', self.reward_samples_debug_history[-1])
              print ('s_Diff_debug_for_L2_loss (scale adjusted):', np.sqrt(self.s_Diff_history[-1])/2 )#check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)
          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()


          
  def state_single_step_gradient(self,s,a):
      """
      input:
          - self.s (T,Ds), the tested state sequence
          - self.a (T,Da), the tested action sequence
          - self.ind
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s
      
      T,_ = np.shape(s)
      ind_hash = np.arange(T-1,dtype = np.int32)
      self.batch_size = T-1
      self.start_line = np.arange(self.batch_size)
      self.memory_frame = 2
      
      minibatch_s = s[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]
      s_init = minibatch_s
#      s_init = minibatch_s[:,:1,:]
#      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = a[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]

      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      self.a_tmp_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_tmp_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate = 0.)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)                           
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(1):
          
          feed = {self.a_tmp_placeholder: minibatch_a}
          Diff_val,s_gds_val = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
#          self.state_decode_gds_list = np.reshape(self.state_decode_gds_list,[-1])
          self.state_decode_loss_listory.append(Diff_val)
          self.state_decode_loss_listory = np.reshape(self.state_decode_loss_listory,[-1])
          
          if t % 100 == 0:
              print (Diff_val)
              
      sess.close()
              

                    
  def state_decode(self,learning_rate = 3e-3,max_iteration = 1000):
      """
      input:
          - self.s (T,Ds), but only the first data is used
          - self.a (T,Da), the tested action sequence
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s          
      
      
      if self.memory_frame < 2:
          print ('Please set memory frame larger than 1')
          return
      batch_size_old = self.batch_size
      ind_hash_old = self.ind_hash
      self.ind_hash = np.arange(1,dtype = np.int32)#currently
      self.batch_size = 1#currently
      self._reset()
      
      
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      s_init = minibatch_s[:,:1,:]
      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
                           
      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      
      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_expert_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.AdadeltaOptimizer(learning_rate)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)
      self.train_op_for_state_decode = cur_optmizer.apply_gradients(self.gds_for_state_decode)
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(max_iteration):
          feed = {self.a_expert_placeholder: minibatch_a}
          Diff_val,s_gds_val,_ = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode,self.train_op_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
          self.state_decode_loss_listory.append(Diff_val)
          if t % 100 == 0:
              print (Diff_val)
          
      self.s_decode_name = [i.name for i in tf.train.variables.trainable_variables()]
      self.s_decode_val =  [i.eval(session = sess) for i in tf.train.variables.trainable_variables()]
      sess.close()
      
      self.batch_size = batch_size_old
      self.ind_hash = ind_hash_old
    
class IRL_Solver_demo19(object):
  """
  residual structure
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.num_of_lms_lines = kwargs.pop('num_of_lms_lines')
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    self.state_decode_only = kwargs.pop('state_decode_only', False)
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1

    

    self.model_params = {}
    self.model_params['hidden_dim'] = self.hidden_size
    self.model_params['s_encode_dim'] = self.hidden_size
    self.model_params['num_of_lms_lines'] = self.num_of_lms_lines
    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

    if self.state_decode_only == False:
        
        self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
        self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
        self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
    #    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    #    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
        self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
        
        
        with tf.variable_scope('reward'):
            self.reward_expert_batch,self.s_Diff_batch,_ = reward_v1_2(self.s_placeholder,self.a_expert_placeholder,self.model_params)
            tf.get_variable_scope().reuse_variables()
            self.reward_policy_batch,_,_ = reward_v1_2(self.s_placeholder_for_policy,self.a_policy_placeholder,self.model_params)
            
    
            
            
            w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
            w = w/tf.reduce_sum(w)
            w = tf.stop_gradient(w)
            
            self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
            self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
    #        self.reward_policy = tf.reduce_mean(reward_policy)
            self.reward_diff = self.reward_expert - self.reward_policy
            self.s_Diff = tf.reduce_mean(self.s_Diff_batch)
    
    
            
        if self.test_only == False:
          
          global_step = tf.Variable(0, trainable=False)
          learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=1,decay_rate=self.optim_config['decay'])
            
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer( learning_rate)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
    #     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
          self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff)
          self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)
    
        
    
            
        self.reward_expert_history = []
        self.reward_policy_history = []
        self.reward_diff_history = []
        self.s_Diff_history = []
        self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
        print (self.tmp3)
        
        localtime = time.asctime(time.localtime(time.time()))
        #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )
        
        weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
        print ('weights_list:', weights_list)
        
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))


    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
#      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, _ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print ('reward_diff:', self.reward_diff_history[-1])

              print ('s_Diff_debug_for_L2_loss:', np.sqrt(self.s_Diff_history[-1])/2) #check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)
          
          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()


          
  def state_single_step_gradient(self,s,a):
      """
      input:
          - self.s (T,Ds), the tested state sequence
          - self.a (T,Da), the tested action sequence
          - self.ind
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s
      
      T,_ = np.shape(s)
      ind_hash = np.arange(T-1,dtype = np.int32)
      self.batch_size = T-1
      self.start_line = np.arange(self.batch_size)
      self.memory_frame = 2
      
      minibatch_s = s[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]
      s_init = minibatch_s
#      s_init = minibatch_s[:,:1,:]
#      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = a[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]

      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      self.a_tmp_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_tmp_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate = 0.)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)                           
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(1):
          
          feed = {self.a_tmp_placeholder: minibatch_a}
          Diff_val,s_gds_val = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
#          self.state_decode_gds_list = np.reshape(self.state_decode_gds_list,[-1])
          self.state_decode_loss_listory.append(Diff_val)
          self.state_decode_loss_listory = np.reshape(self.state_decode_loss_listory,[-1])
          
          if t % 100 == 0:
              print (Diff_val)
              
      sess.close()
              

                    
  def state_decode(self,learning_rate = 3e-3,max_iteration = 1000):
      """
      input:
          - self.s (T,Ds), but only the first data is used
          - self.a (T,Da), the tested action sequence
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s          
      
      
      if self.memory_frame < 2:
          print ('Please set memory frame larger than 1')
          return
      batch_size_old = self.batch_size
      ind_hash_old = self.ind_hash
      self.ind_hash = np.arange(1,dtype = np.int32)#currently
      self.batch_size = 1#currently
      self._reset()
      
      
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      s_init = minibatch_s[:,:1,:]
      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
                           
      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      
      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_expert_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.AdadeltaOptimizer(learning_rate)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)
      self.train_op_for_state_decode = cur_optmizer.apply_gradients(self.gds_for_state_decode)
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(max_iteration):
          feed = {self.a_expert_placeholder: minibatch_a}
          Diff_val,s_gds_val,_ = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode,self.train_op_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
          self.state_decode_loss_listory.append(Diff_val)
          if t % 100 == 0:
              print (Diff_val)
          
      self.s_decode_name = [i.name for i in tf.train.variables.trainable_variables()]
      self.s_decode_val =  [i.eval(session = sess) for i in tf.train.variables.trainable_variables()]
      sess.close()
      
      self.batch_size = batch_size_old
      self.ind_hash = ind_hash_old


class IRL_Solver_demo18(object):
  """
  residual structure
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    self.state_decode_only = kwargs.pop('state_decode_only', False)
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1

    

    self.model_params = {}
    self.model_params['hidden_dim'] = self.hidden_size
    self.model_params['s_encode_dim'] = self.hidden_size
    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

    if self.state_decode_only == False:
        
        self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
        self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
        self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
    #    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    #    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
        self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
        
        
        with tf.variable_scope('reward'):
            self.reward_expert_batch,self.s_Diff_batch,_ = reward_v1_1(self.s_placeholder,self.a_expert_placeholder,self.model_params)
            tf.get_variable_scope().reuse_variables()
            self.reward_policy_batch,_,_ = reward_v1_1(self.s_placeholder_for_policy,self.a_policy_placeholder,self.model_params)
            
    
            
            
            w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
            w = w/tf.reduce_sum(w)
            w = tf.stop_gradient(w)
            
            self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
            self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
    #        self.reward_policy = tf.reduce_mean(reward_policy)
            self.reward_diff = self.reward_expert - self.reward_policy
            self.s_Diff = tf.reduce_mean(self.s_Diff_batch)
    
    
            
        if self.test_only == False:
          
          global_step = tf.Variable(0, trainable=False)
          learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=1,decay_rate=self.optim_config['decay'])
            
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer( learning_rate)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
    #     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
          self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff)
          self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)
    
        
    
            
        self.reward_expert_history = []
        self.reward_policy_history = []
        self.reward_diff_history = []
        self.s_Diff_history = []
        self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
        print (self.tmp3)
        
        localtime = time.asctime(time.localtime(time.time()))
        #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )
    
        
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))


    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
#      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, _ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print ('reward_diff:', self.reward_diff_history[-1])

              print ('s_Diff_debug_for_L2_loss:', np.sqrt(self.s_Diff_history[-1])/2 )#check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()


          
  def state_single_step_gradient(self,s,a):
      """
      input:
          - self.s (T,Ds), the tested state sequence
          - self.a (T,Da), the tested action sequence
          - self.ind
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s
      
      T,_ = np.shape(s)
      ind_hash = np.arange(T-1,dtype = np.int32)
      self.batch_size = T-1
      self.start_line = np.arange(self.batch_size)
      self.memory_frame = 2
      
      minibatch_s = s[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]
      s_init = minibatch_s
#      s_init = minibatch_s[:,:1,:]
#      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = a[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]

      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      self.a_tmp_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_tmp_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate = 0.)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)                           
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(1):
          
          feed = {self.a_tmp_placeholder: minibatch_a}
          Diff_val,s_gds_val = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
#          self.state_decode_gds_list = np.reshape(self.state_decode_gds_list,[-1])
          self.state_decode_loss_listory.append(Diff_val)
          self.state_decode_loss_listory = np.reshape(self.state_decode_loss_listory,[-1])
          
          if t % 100 == 0:
              print (Diff_val)
              
      sess.close()
              

                    
  def state_decode(self,learning_rate = 3e-3,max_iteration = 1000):
      """
      input:
          - self.s (T,Ds), but only the first data is used
          - self.a (T,Da), the tested action sequence
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s          
      
      
      if self.memory_frame < 2:
          print ('Please set memory frame larger than 1')
          return
      batch_size_old = self.batch_size
      ind_hash_old = self.ind_hash
      self.ind_hash = np.arange(1,dtype = np.int32)#currently
      self.batch_size = 1#currently
      self._reset()
      
      
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      s_init = minibatch_s[:,:1,:]
      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
                           
      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      
      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_expert_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.AdadeltaOptimizer(learning_rate)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)
      self.train_op_for_state_decode = cur_optmizer.apply_gradients(self.gds_for_state_decode)
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(max_iteration):
          feed = {self.a_expert_placeholder: minibatch_a}
          Diff_val,s_gds_val,_ = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode,self.train_op_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
          self.state_decode_loss_listory.append(Diff_val)
          if t % 100 == 0:
              print (Diff_val)
          
      self.s_decode_name = [i.name for i in tf.train.variables.trainable_variables()]
      self.s_decode_val =  [i.eval(session = sess) for i in tf.train.variables.trainable_variables()]
      sess.close()
      
      self.batch_size = batch_size_old
      self.ind_hash = ind_hash_old



class IRL_Solver_demo16(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    self.state_decode_only = kwargs.pop('state_decode_only', False)
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1

    

    self.model_params = {}
    self.model_params['hidden_dim'] = self.hidden_size
    self.model_params['s_encode_dim'] = self.hidden_size
    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

    if self.state_decode_only == False:
        
        self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
        self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
        self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
    #    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    #    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
        self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
        
        
        with tf.variable_scope('reward'):
            self.reward_expert_batch,self.s_Diff_batch,_ = reward_v1(self.s_placeholder,self.a_expert_placeholder,self.model_params)
            tf.get_variable_scope().reuse_variables()
            self.reward_policy_batch,_,_ = reward_v1(self.s_placeholder_for_policy,self.a_policy_placeholder,self.model_params)
            
    
            
            
            w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
            w = w/tf.reduce_sum(w)
            w = tf.stop_gradient(w)
            
            self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
            self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
    #        self.reward_policy = tf.reduce_mean(reward_policy)
            self.reward_diff = self.reward_expert - self.reward_policy
            self.s_Diff = tf.reduce_mean(self.s_Diff_batch)
    
    
            
        if self.test_only == False:
          
          global_step = tf.Variable(0, trainable=False)
          learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                               global_step=global_step,
                                               decay_steps=1,decay_rate=self.optim_config['decay'])
            
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer( learning_rate)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
    #     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
    #     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
          self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff)
          self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)
    
        
    
            
        self.reward_expert_history = []
        self.reward_policy_history = []
        self.reward_diff_history = []
        self.s_Diff_history = []
        self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
        print (self.tmp3)
        
        localtime = time.asctime(time.localtime(time.time()))
        #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )
    
        
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))


    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
#      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, _ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print ('reward_diff:', self.reward_diff_history[-1])

              print ('s_Diff_debug_for_L2_loss:', np.sqrt(self.s_Diff_history[-1])/2) #check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)


          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()


          
  def state_single_step_gradient(self,s,a):
      """
      input:
          - self.s (T,Ds), the tested state sequence
          - self.a (T,Da), the tested action sequence
          - self.ind
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s
      
      T,_ = np.shape(s)
      ind_hash = np.arange(T-1,dtype = np.int32)
      self.batch_size = T-1
      self.start_line = np.arange(self.batch_size)
      self.memory_frame = 2
      
      minibatch_s = s[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]
      s_init = minibatch_s
#      s_init = minibatch_s[:,:1,:]
#      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = a[creating_index_matrix(self.start_line,self.memory_frame,ind_hash)]

      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      self.a_tmp_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')

      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_tmp_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate = 0.)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)                           
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(1):
          
          feed = {self.a_tmp_placeholder: minibatch_a}
          Diff_val,s_gds_val = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
#          self.state_decode_gds_list = np.reshape(self.state_decode_gds_list,[-1])
          self.state_decode_loss_listory.append(Diff_val)
          self.state_decode_loss_listory = np.reshape(self.state_decode_loss_listory,[-1])
          
          if t % 100 == 0:
              print (Diff_val)
              
      sess.close()
              

                    
  def state_decode(self,learning_rate = 3e-3,max_iteration = 1000):
      """
      input:
          - self.s (T,Ds), but only the first data is used
          - self.a (T,Da), the tested action sequence
      return:
          - self.state_decode_gds_list [], the gds for state
          - self.state_decode_loss_listory [], the decoded state
      """
      def stop_gradient_of_s0(s):
          s0 = s[:,:1,:]
          s123 = s[:,1:,:]
          s0 = tf.stop_gradient(s0)
          s = tf.concat(1,[s0,s123])
          return s          
      
      
      if self.memory_frame < 2:
          print ('Please set memory frame larger than 1')
          return
      batch_size_old = self.batch_size
      ind_hash_old = self.ind_hash
      self.ind_hash = np.arange(1,dtype = np.int32)#currently
      self.batch_size = 1#currently
      self._reset()
      
      
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      s_init = minibatch_s[:,:1,:]
      s_init = np.tile(s_init,[1,self.memory_frame,1])
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
                           
      init = tf.constant_initializer(value = s_init)
      self.s_variable = tf.get_variable('s_variable',shape=(self.batch_size, self.memory_frame, self.Ds),initializer=init)
      s_tensor = stop_gradient_of_s0(self.s_variable)
      
      _,self.s_Diff_batch_for_state_decode,s_prediction_list = reward_v1(s_tensor, self.a_expert_placeholder, self.model_params,trainable = False)
      self.s_Diff_for_state_decode = tf.reduce_mean(self.s_Diff_batch_for_state_decode)
      
      cur_optmizer = tf.train.AdadeltaOptimizer(learning_rate)
      self.gds_for_state_decode = cur_optmizer.compute_gradients(self.s_Diff_for_state_decode)
      self.train_op_for_state_decode = cur_optmizer.apply_gradients(self.gds_for_state_decode)
      self.state_decode_gds_list = []
      self.state_decode_loss_listory = []

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())
      for t in xrange(max_iteration):
          feed = {self.a_expert_placeholder: minibatch_a}
          Diff_val,s_gds_val,_ = sess.run([self.s_Diff_for_state_decode,self.gds_for_state_decode,self.train_op_for_state_decode],feed_dict = feed)
          self.state_decode_gds_list.append(s_gds_val)
          self.state_decode_loss_listory.append(Diff_val)
          if t % 100 == 0:
              print (Diff_val)
          
      self.s_decode_name = [i.name for i in tf.train.variables.trainable_variables()]
      self.s_decode_val =  [i.eval(session = sess) for i in tf.train.variables.trainable_variables()]
      sess.close()
      
      self.batch_size = batch_size_old
      self.ind_hash = ind_hash_old



class IRL_Solver_demo15(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.load_train = kwargs.pop('load_train', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1
 

    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
    self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    model_params['s_encode_dim'] = self.hidden_size

    with tf.variable_scope('reward'):
        self.reward_expert_batch,self.s_Diff_batch = reward_v01(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        self.reward_policy_batch,_ = reward_v01(self.s_placeholder_for_policy,self.a_policy_placeholder,model_params)
        

        
        
        w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        
        self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
        self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        self.s_Diff = tf.reduce_mean(self.s_Diff_batch) 
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff + self.s_Diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.s_Diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))

    

    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
#      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val, _ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.s_Diff,self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val,s_Diff_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch,self.s_Diff], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.s_Diff_history.append(s_Diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False and self.load_train == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print ('reward_diff:', self.reward_diff_history[-1])

              print ('s_Diff_debug_for_L2_loss:', np.sqrt(self.s_Diff_history[-1])/2 )#check this value with reward_expert_history to estimate scale of gradient
              if self.test_only == False:
                  print ('reward_expert:', self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')

          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

      sess.close()
      
      
      
      
    
class IRL_Solver_demo13(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1
 

    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
    self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    model_params['s_encode_dim'] = self.hidden_size

    with tf.variable_scope('reward'):
        self.reward_expert_batch,s_Diff = reward_v01(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        self.reward_policy_batch,s_Diff = reward_v01(self.s_placeholder_for_policy,self.a_policy_placeholder,model_params)
        

        
        
        w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        
        self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
        self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))

    

    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
#      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
#              print self.reward_expert_history[-1]
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')

          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

          

      sess.close()
      

class IRL_Solver_demo12(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, ind_hash, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.ind_hash = ind_hash
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1
 

    self.data_index_range = self.ind_hash.shape[0] - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.Ds),name = 's')#(N,D)
    self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.memory_frame, self.Ds),name = 's_sampling')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.memory_frame, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy,  self.memory_frame,  self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    model_params['s_encode_dim'] = self.hidden_size

    with tf.variable_scope('reward'):
        self.reward_expert_batch,s_Diff = reward_v01(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        self.reward_policy_batch,s_Diff = reward_v01(self.s_placeholder_for_policy,self.a_policy_placeholder,model_params)
        

        
        
        w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        
        self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
        self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def a_sampling_uniform_v02(self,N,range_list,memory_frame):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
#      ????????????????????????
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)
    
    data_all = self.a[creating_index_matrix_without_ind(self.ind_hash,self.memory_frame)]
    data_all = np.reshape(data_all,[-1,self.Da * self.memory_frame])
    
    self.kde_generator = KernelDensity(kernel='tophat', bandwidth=0.002).fit(data_all)
    self.kde_estimator = KernelDensity(kernel='tophat', bandwidth=0.05).fit(data_all)

#    samples = kde.sample(self.batch_size_for_policy)
#    sanples_w = np.exp(kde2.score_samples(samples))

    

    
  def _step(self, sess = None):
      
      samples = self.kde_generator.sample(self.batch_size_for_policy)
      policy_a = np.reshape(samples, (self.batch_size_for_policy,  self.memory_frame,  self.Da)  )
#      samples_w = np.exp(self.kde_estimator.score_samples(samples))#still...too...slow
      samples_w = np.ones(self.batch_size_for_policy)
      
#      minibatch_s = self.s[self.ind_hash[self.start_line]]
      minibatch_s = self.s[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1,1])
#      minibatch_a = self.a[self.ind_hash[self.start_line]]
      minibatch_a = self.a[creating_index_matrix(self.start_line,self.memory_frame,self.ind_hash)]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: samples_w
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
#              print self.reward_expert_history[-1]
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          

          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

          

      sess.close()
      
        

class IRL_Solver_demo10(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = 1
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    
    if self.test_only == True:
        self.iteration = s.shape[0]/self.batch_size+1
 

    self.data_index_range = self.s.shape[0] - (self.memory_frame-1) * self.step_size - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.Ds),name = 's_sampling')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy, self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    with tf.variable_scope('reward'):
        self.reward_expert_batch = reward_v02(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        self.reward_policy_batch = reward_v02(self.s_placeholder_for_policy,self.a_policy_placeholder,model_params)
        

        
        
        w = tf.exp(self.reward_policy_batch) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        
        self.reward_policy = tf.reduce_sum(self.reward_policy_batch * w)
        self.reward_expert = tf.reduce_mean(self.reward_expert_batch)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    else:
        tmp = 0
        self.start_line = np.arange(self.batch_size)
    self.start_line = np.mod(self.start_line,self.data_index_range)

    
  def _step(self, sess = None):
      
      


      policy_a = self.a_sampling_uniform_v01(self.batch_size_for_policy,self.sampling_range_list)
      minibatch_s = self.s[self.start_line]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1])
      minibatch_a = self.a[self.start_line]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: np.ones(self.batch_size_for_policy)
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert_batch,self.reward_diff, self.reward_policy_batch], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      if self.test_only == False:
          self.start_line += 1
      else:
          self.start_line += self.batch_size
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
              print (self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          

          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)
      else:
          self.reward_expert_history = np.reshape(self.reward_expert_history,[-1])
          self.reward_expert_history = self.reward_expert_history[:self.data_index_range]
          self.reward_diff_history = np.reshape(self.reward_diff_history,[-1])
          self.reward_diff_history = self.reward_diff_history[:self.data_index_range]

          

      sess.close()
      
        
        
        
class IRL_Solver_demo6(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = 1
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.batch_size = kwargs.pop('batch_size', 100)
    sample_expert_ratio = kwargs.pop('sample_expert_ratio', 1)
    self.sample_expert_ratio = sample_expert_ratio
    self.batch_size_for_policy = self.batch_size * sample_expert_ratio
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo6/')
    
    if self.test_only == True:
        self.batch_size = 1
        self.sample_expert_ratio = 1
        self.batch_size_for_policy = 1
        self.iteration = s.shape[0]
 

    self.data_index_range = self.s.shape[0] - (self.memory_frame-1) * self.step_size - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    self.s_placeholder_for_policy = tf.placeholder(tf.float32, (self.batch_size_for_policy, self.Ds),name = 's_sampling')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size_for_policy, self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size_for_policy))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    with tf.variable_scope('reward'):
        reward_expert = reward_v02(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        reward_policy = reward_v02(self.s_placeholder_for_policy,self.a_policy_placeholder,model_params)
        self.reward_expert = tf.reduce_mean(reward_expert)
        

        
        
        w = tf.exp(reward_policy) / tf.reshape(self.sampling_distribution,(self.batch_size_for_policy,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        self.reward_policy = tf.reduce_sum(reward_policy * w)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
    else:
        tmp = 0
    tmp2 = self.data_index_range / self.batch_size
    self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    self.start_line = np.mod(self.start_line,self.data_index_range)


    
  def _step(self, sess = None):
      
      


      policy_a = self.a_sampling_uniform_v01(self.batch_size_for_policy,self.sampling_range_list)
      minibatch_s = self.s[self.start_line]
      minibatch_s_for_policy = np.tile(minibatch_s,[self.sample_expert_ratio,1])
      minibatch_a = self.a[self.start_line]
      feed = {self.s_placeholder: minibatch_s,
              self.s_placeholder_for_policy: minibatch_s_for_policy,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: np.ones(self.batch_size_for_policy)
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert,self.reward_diff, self.reward_policy], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      self.start_line += 1
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
              print (self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          

          
          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)
          summary_writer.add_graph(sess.graph)

          

      sess.close()
      
        
        
class IRL_Solver_demo2(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = 1
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.batch_size = kwargs.pop('batch_size', 100)
    self.policy_batch_size = kwargs.pop('policy_batch_size', self.batch_size)
    self.sampling_range_list = kwargs.pop('sampling_range_list', np.array([[0,0],[1.,4.]]))
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/demo2/')
    
    if self.test_only == True:
        self.batch_size = 1
        self.iteration = s.shape[0]
 

    self.data_index_range = self.s.shape[0] - (self.memory_frame-1) * self.step_size - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    with tf.variable_scope('reward'):
        reward_expert = reward_v02(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        reward_policy = reward_v02(self.s_placeholder,self.a_policy_placeholder,model_params)
        self.reward_expert = tf.reduce_mean(reward_expert)
        w = tf.exp(reward_policy) / tf.reshape(self.sampling_distribution,(self.batch_size,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        self.reward_policy = tf.reduce_sum(reward_policy * w)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  + range_list[0]
      return a
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
    else:
        tmp = 0
    tmp2 = self.data_index_range / self.batch_size
    self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    self.start_line = np.mod(self.start_line,self.data_index_range)


    
  def _step(self, sess = None):
      
      


      policy_a = self.a_sampling_uniform_v01(self.batch_size,self.sampling_range_list)
      minibatch_s = self.s[self.start_line]
      minibatch_a = self.a[self.start_line]
      feed = {self.s_placeholder: minibatch_s,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: np.ones(self.batch_size)
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert,self.reward_diff, self.reward_policy], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      self.start_line += 1
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
              print (self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          


          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)

          

      sess.close()
      

class IRL_Solver(object):
  """
  single-step version
  """
    
  def __init__(self, s, a, **kwargs):
    """
    model:reward
    s,a:(N,Ds),(N,Da)
    """
    self.dipict = 'v02'
    self.model = object

#    self.model.savePath = ''
    self.memory_frame = 1
    self.window_size = 1
    self.s = s
    self.a = a
    self.Ds = s.shape[1]
    self.Da = a.shape[1]
    self.batch_size = kwargs.pop('batch_size', 100)
    #self.batch_size = kwargs.pop('batch_size', 100)
    self.memory_frame = kwargs.pop('memory_frame',1)
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.test_only = kwargs.pop('test_only', False)
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.step_size = kwargs.pop('step_size', 1)
    self.print_every = kwargs.pop('print_every', 20)
    self.iteration = kwargs.pop('iteration', 500)
    self.hidden_size = kwargs.pop('hidden_size', 50)
    self.save_dir = kwargs.pop('save_dir', './save/1/')
    
    if self.test_only == True:
        self.batch_size = 1
        self.iteration = s.shape[0]
 

    self.data_index_range = self.s.shape[0] - (self.memory_frame-1) * self.step_size - (self.window_size-1) * self.step_size 

    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
    self.a_expert_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_expert')
    self.a_policy_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a_policy')
#    self.s_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.Ds),name = 's')#(N,D)
#    self.a_placeholder = tf.placeholder(tf.float32,(self.batch_size, self.Da),name = 'a')
    self.sampling_distribution = tf.placeholder(tf.float32,(self.batch_size))
    

    model_params = {}
    model_params['hidden_dim'] = self.hidden_size
    with tf.variable_scope('reward'):
        reward_expert = reward_v02(self.s_placeholder,self.a_expert_placeholder,model_params)
        tf.get_variable_scope().reuse_variables()
        reward_policy = reward_v02(self.s_placeholder,self.a_policy_placeholder,model_params)
        self.reward_expert = tf.reduce_mean(reward_expert)
        w = tf.exp(reward_policy) / tf.reshape(self.sampling_distribution,(self.batch_size,1))
        w = w/tf.reduce_sum(w)
        w = tf.stop_gradient(w)
        self.reward_policy = tf.reduce_sum(reward_policy * w)
#        self.reward_policy = tf.reduce_mean(reward_policy)
        self.reward_diff = self.reward_expert - self.reward_policy
        
    if self.test_only == False:
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(self.optim_config['learning_rate'],
                                           global_step=global_step,
                                           decay_steps=1,decay_rate=self.optim_config['decay'])
        
      if(self.update_rule == 'adam'):
#        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.AdamOptimizer( learning_rate)
      elif(self.update_rule == 'sgd'):
#        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
        cur_optmizer = tf.train.GradientDescentOptimizer(learning_rate)
#     self.gds_expert = cur_optmizer.compute_gradients(-self.reward_expert)
#     self.gds_policy = cur_optmizer.compute_gradients(self.reward_policy)
      self.gds = cur_optmizer.compute_gradients(-self.reward_diff)
      self.train_op = cur_optmizer.apply_gradients(self.gds,global_step = global_step)

    

        
    self.reward_expert_history = []
    self.reward_policy_history = []
    self.reward_diff_history = []
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)
    
    localtime = time.asctime(time.localtime(time.time()))
    #self.save_dir = ('./save/%s_%ddataFrame_%dstep_size_%dbatchsize_%siteration_%shiddenSize_%s/' %(self.dipict , s.shape[0], self.step_size, self.batch_size, self.iteration,model_params['hidden_dim'], localtime) )

    
  def a_sampling_uniform_v01(self,N,range_list):
      """
      range_list:(2,Da) pair of 'mean,range'
      """
      _,Da = np.shape(range_list)
      a =  (np.random.rand(N,Da)-0.5)*range_list[1]  - range_list[0]
      return a
      
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.test_only == False:
        tmp = np.random.randint(self.data_index_range)
    else:
        tmp = 0
    tmp2 = self.data_index_range / self.batch_size
    self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
    self.start_line = np.mod(self.start_line,self.data_index_range)


    
  def _step(self, sess = None):
      
      

      #range_list = np.array([[8,0],[0,0.1]])#([mean,mean],[range,range])
      range_list = np.array([[0],[4]])
      policy_a = self.a_sampling_uniform_v01(self.batch_size,range_list)
      minibatch_s = self.s[self.start_line]
      minibatch_a = self.a[self.start_line]
      feed = {self.s_placeholder: minibatch_s,
              self.a_expert_placeholder: minibatch_a,
              self.a_policy_placeholder: policy_a,
              self.sampling_distribution: np.ones(self.batch_size)
              }
              
      
      if self.test_only == False:
          reward_expert_val,reward_diff_val,reward_policy_val,_ = sess.run([self.reward_expert,self.reward_diff, self.reward_policy, self.train_op], feed_dict = feed)
      else:
          reward_expert_val,reward_diff_val,reward_policy_val = sess.run([self.reward_expert,self.reward_diff, self.reward_policy], feed_dict=feed)
      self.reward_diff_history.append(reward_diff_val)
      self.reward_expert_history.append(reward_expert_val)
      self.reward_policy_history.append(reward_policy_val)
      self.start_line += 1
      self.start_line = np.mod(self.start_line,self.data_index_range)
      
  def train(self):
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.test_only == False:
          init = tf.global_variables_initializer()
          sess.run(init)
      else:
          tf.train.Saver().restore(sess, self.save_dir+'params')
      
      self._reset()
      
      for t in xrange(self.iteration):
          self._step(sess)
          
          if t % self.print_every == 0:
              print ('(Iteration %d / %d) : ' % (t + 1, self.iteration))
              print (self.reward_diff_history[-1])
              print (self.reward_expert_history[-1])
      if self.test_only == False:
          try:
              os.mkdir(self.save_dir)
          except:
              pass
          saver.save(sess, self.save_dir + 'params')
          

          #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
          summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

          summary_writer.add_graph(sess.graph)

          

      sess.close()
    
    
class Simulation_Solver(object):
  """
  modified from RNN_Solver;
  label_step_delay and zero_point_of_window are deprecated
  IN A WORDS: INPUT AND GTS ARE SYNCHRONOUS
  
  new config params: self.window_size
  
  data_index_range and  self.start_line are hence moldified
  """

  def __init__(self, model, data, **kwargs):
    """
    Construct a new CaptioningSolver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data from load_coco_data

    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    self.model = model
    self.implement = self.model.implement
    self.data = data
    self.Din = data['input'].shape[1]
    self.dataFrame = data['input'].shape[0]
    self.Dout = data['groundtruth'].shape[1]
    
    # Unpack keyword arguments
    self.step_size =  kwargs.pop('step_size',1)#training data stride
    self.start_point_range = kwargs.pop('start_point_range',1)  # start point of sequence < this      
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)
    self.depict = kwargs.pop('depict', 'simulationLoss')
    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)
    self.load_model = kwargs.pop('load_model',False)
    self.only_test_model = kwargs.pop('only_test_model',False)
    self.memory_frame = kwargs.pop('memory_frame',30)
    self.evaluation_mode = kwargs.pop('evaluation_mode',False)
    self.reset_when_crash = kwargs.pop('reset_when_crash',False)
    if self.only_test_model == True:
        if self.load_model == False:
            raise NotImplementedError
#        self.memory_frame = 1#no need to do BPTT when test
        self.num_epochs = 1
    #new
    self.circle_batch_level = kwargs.pop('circle_batch_level', 0)#0:pure line;  1:startline mod index_range  2:not implement
    #for gm
    self.window_size = kwargs.pop('window_size', 3)
    self.simu_init_flag = kwargs.pop('simu_init_flag',0)
    self.loss_level = kwargs.pop('loss_level','acc')

    
    if self.implement == 'tensorflow':#tensorflow 'config' part
      print ('--using tensorflow--')
      if self.load_model == False:
          tmp = str(self.model.tfmodel.state_size)
          localtime = time.asctime( time.localtime(time.time()) )
          self.save_dir = './save/%s_%s_%ddataFrame_%slossLevel_%dsimu_init_flag_%dstep_size_%dmemoryFrame_%dbatchsize_%dwindowSize_%depoch_%shiddenSize_%s' %(self.depict ,self.model.depict , self.dataFrame, self.loss_level, self.simu_init_flag, self.step_size,self.memory_frame, self.batch_size, self.window_size, self.num_epochs, tmp, localtime)
          os.mkdir(self.save_dir)
          self.savePath = '%s/weight' %self.save_dir
          self.model.savePath = self.savePath  
          self.model.save_dir = self.save_dir
          self.model.solver_step_size = self.step_size
          self.model.solver_window_size = self.window_size
      else:
          self.save_dir = self.model.save_dir
          self.savePath = self.model.savePath
          self.step_size = self.model.solver_step_size
          self.window_size = self.model.solver_window_size

      #self.state_holder = self.model.tfmodel.zero_state(self.batch_size,dtype = tf.float32)
      self.input_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.window_size, self.Din),name = 'input')#(N,T,W,D)
      self.labels_placeholder = tf.placeholder(tf.float32, (self.batch_size, self.memory_frame, self.window_size, self.Dout),name = 'label')
      self.ini_state_prev = self.model.generate_simustates_prev(self.input_placeholder)
      self.simu_env_holder = self.ini_state_prev
      self.tfloss, self.outputs, self.states, self.info = self.model.loss(self.input_placeholder,self.labels_placeholder,self.simu_env_holder,simu_init_flag = self.simu_init_flag,loss_level = self.loss_level,test_mode =self.evaluation_mode, reset_when_crash=self.reset_when_crash)
      #outputs: T x (N,D)
      if self.only_test_model == False:
          if(self.update_rule == 'adam'):
    #        self.train_op = tf.train.AdamOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.AdamOptimizer(self.optim_config['learning_rate'])
            self.gds = cur_optmizer.compute_gradients(self.tfloss)
            self.train_op = cur_optmizer.apply_gradients(self.gds)
          elif(self.update_rule == 'sgd'):
    #        self.train_op = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate']).minimize(self.tfloss)
            cur_optmizer = tf.train.GradientDescentOptimizer(self.optim_config['learning_rate'])
            self.gds = cur_optmizer.compute_gradients(self.tfloss)
            self.train_op = cur_optmizer.apply_gradients(self.gds)
    else:
        raise NotImplementedError
      
    l_input = self.data['input'].shape[0]
    self.data_index_range = (l_input - (self.memory_frame-1) * self.step_size - (self.window_size-1) * self.step_size ) #start point index should < this
    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())

      raise ValueError('Unrecognized arguments %s' % extra)
    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)


    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.outputs_history = []
    self.gt_history = []
    self.vT_history = []
    self.disTgt_history = []
    self.disSim_history = []
    self.vSim_history = []
    self.gds_history = []
    self.train_acc_history = []
    self.val_acc_history = []
    self.one_gds = 0
    
    #add 'window' dim    
    self.tmp3 = creat_index_bias_matrix3(self.step_size,self.memory_frame, self.window_size)
    print (self.tmp3)


    self._reset()




  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """

    if self.circle_batch_level == 0:
        #self.start_line : (N,) like (9,10,11,12,13,14,15,0,1,2,3,...)
        rand_start_point = np.random.randint(self.start_point_range)
        self.start_line = np.mod(np.arange(self.batch_size) + rand_start_point, self.memory_frame * self.step_size)
        self.start_line = np.mod(self.start_line,self.data_index_range)
#    print self.start_line    
    elif self.circle_batch_level == 1:
        tmp = np.random.randint(self.step_size)
        tmp2 = self.data_index_range / self.batch_size
        self.start_line = np.arange(tmp, tmp+tmp2*self.batch_size, tmp2)
        self.start_line = np.mod(self.start_line,self.data_index_range)
        

    batch_window_index = creat_batch_window_index(self.tmp3,self.start_line)
    minibatch = self.data['input'][batch_window_index]  #(N,T,Window,Din)
    with tf.Session() as tmpSess:
        self.states_prev = tmpSess.run(self.ini_state_prev, feed_dict = {self.input_placeholder:minibatch}) #for feed


#I think it's no ever important to do as below:        
#    elif self.implement == 'tensorflow':
#        self.states_prev = self.model.tfmodel.zero_state(self.batch_size,dtype = tf.float32)





  def _step(self,sess = None,test_only = False):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    """
    # Make a minibatch of training data
    
#    minibatch_index = self.tmp + self.start_line   #(T,N) + (N,)
#    batch_window_index = self.tmp2[minibatch_index.T]  #N,T,Window
    

    batch_window_index = creat_batch_window_index(self.tmp3,self.start_line)
#    print batch_window_index
    minibatch = self.data['input'][batch_window_index]  #(N,T,Window,Din)
    batchlabel = self.data['groundtruth'][batch_window_index] #(N,T,Window,Dout)
    
    # Compute loss and gradient, then optimize
    if self.implement == 'tensorflow':    #model def
        feed = {self.input_placeholder: minibatch,
                self.labels_placeholder: batchlabel,
                self.simu_env_holder: self.states_prev,
                }
        if test_only == False:
            loss, self.states_prev, _, self.one_gds, _info ,self.aT = sess.run([self.tfloss, self.states, self.train_op, self.gds ,self.info,self.outputs], feed_dict=feed)
#            loss, self.states_prev, self.one_gds, _info ,self.aT = sess.run([self.tfloss, self.states, self.gds ,self.info,self.outputs], feed_dict=feed) 
        else:
            loss, self.states_prev, _info ,self.aT = sess.run([self.tfloss, self.states, self.info,self.outputs], feed_dict=feed)
            aTgt,vTgt,disTgt,disSim,vSim = _info[-1]
            self.gt_history.append(aTgt)#memory frame must be one
            self.vT_history.append(vTgt)
            self.disTgt_history.append(disTgt)
            self.disSim_history.append(disSim)
            self.vSim_history.append(vSim)
        self.outputs_history.append(self.aT)
    else:
        raise NotImplementedError

    self.loss_history.append(loss)
    self.gds_history.append(self.one_gds)
    #prepare next minibatch
    self.start_line += self.step_size * self.memory_frame
    
    if self.circle_batch_level == 1:
        
        
        
        self.start_line = np.mod(self.start_line,self.data_index_range)

  
  # TODO: This does nothing right now; maybe implement BLEU?
  
  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    """
    Not implemented;
    Check accuracy of the model on the provided data.
    
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,)
    - num_samples: If not None, subsample the data and only test the model
      on num_samples datapoints.
    - batch_size: Split X and y into batches of this size to avoid using too
      much memory.
      
    Returns:
    - acc: Scalar giving the fraction of instances that were correctly
      classified by the model.
    """
    return 0.0
    
    # Maybe subsample the data
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Compute predictions in batches
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
    """
    Run optimization to train the model.
    """
    test_only = False
    if self.load_model == True:
        if self.only_test_model == True:
            test_only = True
            print ('-test only-')

    if self.data_index_range - 1 - self.start_point_range <= 0:
        print ('data too short or start_point_range too big')
    
    if self.batch_size > self.memory_frame*self.step_size  and self.circle_batch_level == 0:
        print ('warning: self.batch_size > self.memory_frame*self.step_size, training data repeated in batch')
    
    #some to initialize
    self._reset()

#    iterations_per_epoch = 1+np.int((self.data_index_range - 1 - self.start_point_range - (self.window_size-1)*self.step_size )/ (self.memory_frame * self.step_size) )    
    iterations_per_epoch = np.int((self.data_index_range - 1 - self.start_point_range - (self.window_size-1)*self.step_size )/ (self.memory_frame * self.step_size) )
    num_iterations = self.num_epochs * iterations_per_epoch
    
    sess = None
    if self.implement == 'tensorflow':
      saver = tf.train.Saver()
      sess = tf.Session()
      if self.load_model == False:
        init = tf.global_variables_initializer()
        sess.run(init)
      else:
        tf.train.Saver().restore(sess, self.model.savePath)

    for t in xrange(num_iterations):
      if np.mod(t,iterations_per_epoch) == 0:  #some to ini and reset
        self._reset()

      if self.implement == 'tensorflow':
        self._step(sess,test_only)

      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print ('(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1]))


      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        
    if self.implement == 'tensorflow':
      if test_only == False:
        saver.save(sess, self.savePath)
        
        #summary_writer = tf.train.SummaryWriter('/tmp/tf_log',sess.graph)
        summary_writer = tf.summary.FileWriter('/tmp/tf_log',sess.graph)

        summary_writer.add_graph(sess.graph)
      sess.close()

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      # TODO: Implement some logic to check Bleu on validation set periodically

    # At the end of training swap the best params into the model
    # self.model.params = self.best_params
  def input_analyze(self, data, hprev = None, cprev = None):
    """
    Analyze runtime importance of input dimension using bp.
    Only for numpy implement
    
    data formation:
    - data.input (N,Din)
    - data.groundtruth (N,Dout)
    """
    if hprev == None:
      hprev = np.zeros([1, self.model.params['Wy'].shape[0]])
    if cprev == None:
      cprev = np.zeros_like(hprev)
        
    sample_input = np.array([data['input']])
    input_idx = np.arange(0,np.size(data['input'],axis = 0)- self.label_step_delay ,self.step_size)
    sample_input = sample_input[:,input_idx,:]
    
    sample_output = np.array([data['groundtruth']])
    output_idx = input_idx + self.label_step_delay
    sample_output = sample_output[:,output_idx,:]
        
        
    loss, grads, h_new, c_new = self.model.loss(sample_input, sample_output, hprev, cprev)
    dx_history = grads['x']
    
#    #interp
#    if self.step_size > 1:
#      sample_idx_interp = np.arange(0,np.size(data['input'],axis = 0))
#      dx_history_interp = np.interp(sample_idx_interp,input_idx,dx_history)
    
    
    
    
    return dx_history