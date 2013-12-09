import sys
import time
from batch_creator import batch_creator
sys.path.append('C:/cudamat/')
import gnumpy as gpu
import numpy as np
from util import Util
import gc


class RBM:
    '''Stacked multilayer restricted Boltzmann machine trained with 
       contrastive divergence.    
    '''
    
    def __init__(self, data, weights, epochs, is_sparse = False, GPU = 0, activation = gpu.logistic, use_noise = False, logger = None):
        self.u = Util()
        gpu.board_id_to_use = GPU
        print 'USE GPU' + str(gpu.board_id_to_use)
        gpu.expensive_check_probability = 0       
        self.func = activation
        
        self.is_sparse = is_sparse
        self.logger = logger
        
        self.batch_size = 100
        self.current_batch_size = 100
     
        self.alpha = 0.1    
        self.epochs = epochs
        
        self.log_message(self.alpha)        
       
        self.X = data
            
        self.input_original = None
        
        self.input = data.shape[1]   
        self.time_interval = 0
        self.weights_to_do = weights
        self.trained_weights = []
        
    def log_message(self, s):
        if self.logger:
            self.logger.info(s)
        print s
        
        
    def allocate_batch(self, start_idx):                        
        end_idx = (start_idx)+self.batch_size 
        end_idx = self.X.shape[0] if end_idx >= self.X.shape[0] else end_idx
                 
        if start_idx >= end_idx: 
            return True          
        self.batch = gpu.garray(self.X[start_idx:end_idx,:] if not self.is_sparse else self.X[start_idx:end_idx,:].todense())   
        
        self.current_batch_size = self.batch.shape[0]
             
        return start_idx >= end_idx


    def positive_phase(self):     
        self.h = self.func(gpu.dot(self.input_dropped,self.w)+self.bias_h) 
        self.w_updt += gpu.dot(self.input_dropped.T,self.h)        
        self.bias_h_updt += gpu.sum(self.h,axis=0)
        self.bias_v_updt += gpu.sum(self.input_dropped,axis=0)          
        
    def gibbs_updates(self, weight_size): 
        self.h = (self.h > gpu.rand(self.current_batch_size,weight_size))       
        
    def negative_phase(self):    
        self.input_dropped = self.func(gpu.dot(self.h,self.w.T)+self.bias_v)
        self.h = self.func(gpu.dot(self.input_dropped,self.w)+self.bias_h)
        self.w_updt -= gpu.dot(self.input_dropped.T,self.h)
        self.bias_h_updt -= gpu.sum(self.h,axis=0)
        self.bias_v_updt -= gpu.sum(self.input_dropped,axis=0)
        
    def get_visible_vector(self, batch):
        if len(self.trained_weights) > 0:       
            visible_vector = batch          
            for weight_pair in self.trained_weights:            
                visible_vector = self.func(gpu.dot(visible_vector, weight_pair[0]) + weight_pair[1])
            return visible_vector
        else:
            return batch

    def initialize_weights(self, size):
        self.w = gpu.garray(np.random.randn(self.input,size))*0.1        
        self.bias_h = gpu.zeros((1,size))
        self.bias_v = gpu.zeros((1,self.input))
        self.w_updt = gpu.zeros((self.input, size))
        self.bias_h_updt = gpu.zeros((1,size))
        self.bias_v_updt = gpu.zeros((1,self.input))
        self.h = gpu.zeros((100,size))
        self.input_dropped = gpu.zeros((100,self.input))
        
        
    def free_GPU_memory(self):
        self.w = 0    
        self.bias_h = 0
        self.bias_v = 0
        self.w_updt = 0
        self.bias_h_updt = 0
        self.bias_v_updt =0
        self.h = 0
        self.input_dropped = 0
        self.X = 0
        gc.collect()

    def train(self):
        self.time_interval = 0
        t1 = time.time()
        cd = 1
        for current_epochs, weight_size in zip(self.epochs, self.weights_to_do):               
            self.initialize_weights(weight_size)   
            for epoch in xrange(current_epochs):    
                error = 0   
                for start_idx in range(0,self.X.shape[0],self.batch_size):                       
                    self.w_updt = gpu.zeros((self.input, weight_size))
                    self.bias_h_updt = gpu.zeros((1,weight_size))
                    self.bias_v_updt = gpu.zeros((1,self.input)) 
                     
                    self.allocate_batch(start_idx)    
                    self.input_original = self.get_visible_vector(self.batch)
                    self.input_dropped = self.input_original                                    
                    self.positive_phase()
                    self.gibbs_updates(weight_size)
                    for j in range(cd):                    
                        self.negative_phase()                
                
                    self.w += self.alpha*self.w_updt/float(self.current_batch_size)
                    self.bias_h += self.alpha*self.bias_h_updt/float(self.current_batch_size)
                    self.bias_v += self.alpha*self.bias_v_updt/float(self.current_batch_size)
                    t0 = time.time()
                    error += gpu.mean((self.input_dropped-self.input_original)**2)
                    self.time_interval += time.time() - t0
                    
                s = 'EPOCH: ' + str(epoch + 1)
                self.log_message(s)
                s = 'Reconstruction error: ' + str(error/(self.X.shape[0]/float(self.batch_size)))
                self.log_message(s)
                
            self.trained_weights.append([self.w.as_numpy_array(), self.bias_h.as_numpy_array()])
            self.input = self.w.shape[1]
            
                    
          
        print 'Time interval: ' + str(self.time_interval)
        print 'Training time: ' + str(time.time() - t1)            
        
        self.free_GPU_memory()
            
        return self.trained_weights
        
        
        
        
        
        
