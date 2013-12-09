import sys
import time
import os, io
from batch_creator import batch_creator
sys.path.append('C:/cudamat/')
import gnumpy as gpu
import numpy as np
from util import Util
import msvcrt
import logging as log
import copy
from scipy import sparse
from sklearn import metrics
from gnumpy_RBM import RBM
from SDA import SDA
import operator
import time

gpu.board_id_to_use = 0
print 'USING GPU' + str(gpu.board_id_to_use)
gpu.expensive_check_probability = 0
log.basicConfig(filename='C:/net/log_GPU{0}.txt'.format(gpu.board_id_to_use), format='%(message)s', level=log.DEBUG)
gpu.seed_rand(1234)

class Deep_net:
    
    def __init__(self, hidden_sizes, cost = None, dropout = [0.2,0.5], set_sizes = [0.8,0.2,0], epochs = 500, is_sparse = False,
                 problem = 'classification',  data = None, printing = True,
                 pretraining_weights = None, pretraining = None, gpu_data = True, ids = None, comment = None, save_weights = False,
                 learning_rate = None, time_printing = False, file_id = '', stop_at_score = None):
        
        log.info("--------------------------------------------")
        log.info("DEEP NEURAL NETWORK")
        log.info("--------------------------------------------")
        log.info('')
        
        
        self.path = 'C:/net/'
        self.util = Util()
        self.rng = np.random.RandomState(1234)   
        self.is_sparse = is_sparse
        self.problem = problem   
        self.printing = printing   
        self.gpu_data = gpu_data
       
        self.ids = ids
        self.create_prediction = False
        self.create_cv_predictions = True
        self.comment = comment
        self.save_weights = save_weights
        self.time_printing = time_printing
        self.file_id = file_id
        self.stop_at_score = stop_at_score
        self.cost = cost
               
        self.allocate_data(data)                  
        self.set_functions_for_problem()
        
        if self.cost == 'auc':
            self.best_result_type = 'increasing'
        else:
            self.best_result_type = 'decreasing'
            
        self.last_train_error = 1 if self.best_result_type == 'decreasing' else 0  
                   
        if pretraining != None:
            pretraining_weights = self.pretrain_weights(hidden_sizes, pretraining)       
       
        self.set_layer_sizes(hidden_sizes)
        self.create_weights(pretraining_weights)              
                        
        self.dropout = dropout
        self.set_sizes = set_sizes
        self.set_names = ['train','cv','test', 'train_error'] 
        
        end_train = np.int(np.round(self.X.shape[0]*self.set_sizes[0]))
        end_cv = np.int(np.round(self.X.shape[0]*(self.set_sizes[0]+ self.set_sizes[1])))
                  
        self.set_boundaries = {'train':end_train, 'cv':end_cv, 'test':self.X.shape[0], 'train_error':end_train }
        self.set_sizes_absolute = {'train':end_train, 'cv':end_cv - end_train, 'test':self.X.shape[0] - end_cv, 'train_error':end_train}   
                       
        self.batch_size = 100
        self.epochs = epochs
        self.learning_rate = 0.1 if learning_rate == None else learning_rate
        self.initial_learning_rate = self.learning_rate 
        self.momentum = 0.5        
        self.momentum_type = 1
        self.transition = 70
        self.end_momentum = 0.95
        self.L2 = 0
        self.L1 = 0
        self.clip_values = 1
        
        self.log_params()
        
        self.set_error_by_epoch = {'train_error':[], 'test':[], 'cv':[]}
        
        self.best_result = [0,0] if self.cost == 'auc' or problem != 'regression' else [0,1] 
        
        self.results = {}
        self.time_values = {} 
        self.train()  
           
    
    def print_and_log(self, s):
        if self.printing:
            print s
        log.info(s) 
        
    def log_params(self):
        self.print_and_log(self.batch_size)
        self.print_and_log(self.epochs)
        self.print_and_log(self.learning_rate)
        self.print_and_log(self.momentum)
        self.print_and_log(self.momentum_type)
        self.print_and_log(self.transition)
        self.print_and_log(self.end_momentum)
        self.print_and_log(self.L2)
        self.print_and_log(self.L1)
        self.print_and_log(self.comment)
        
    def timer_logger(self, name, time_value):
        if name not in  self.time_values.keys():
            self.time_values[name] = [time_value, 0]
        elif self.time_values[name][0] != 0:
            timespan = time_value - self.time_values[name][0]
            old_timespan =  self.time_values[name][1]
            self.time_values[name] = [0,np.round(timespan+old_timespan,2)]
        else:            
            self.time_values[name] = [time_value,self.time_values[name][1]]
            
    def print_timelog(self):               
        time_pairs = sorted(self.time_values.iteritems(), key=operator.itemgetter(1), reverse=True)
        for pair in time_pairs:
            if pair[1][1] > 0.5:
                self.print_and_log('{0}: {1} seconds'.format(pair[0], pair[1][1]))
        
    def set_functions_for_problem(self):
        ''' sets the specific activation functions and their gradients for hidden units and output units  
        '''
        if self.problem == 'classification':
            #logistic + softmax + cross entropy for classification
            self.activation = gpu.logistic
            self.output = gpu.softmax
            self.activation_gradient = lambda X: X*(1-X)
        if self.problem == 'regression':
            #rectified linear + linear + squared error
            self.activation = lambda X: X*(X >= 0)
            self.activation_gradient = lambda X: X > 0
            self.output = lambda X: X
        #add new "problem" with new activation functions here
         
               
    def allocate_data(self, data):         
        if self.gpu_data:
            self.print_and_log('allocating data on the gpu...')
            self.X = gpu.garray(data[0] if not self.is_sparse else data[0].todense())      
            if data[2] != None:     
                self.test = gpu.garray(data[2] if not self.is_sparse else data[2].todense())
            else:
                self.test = gpu.zeros((1,1))
        else:
            self.X = data[0]        
            self.test = data[2]
            
        self.y = data[1] 
        pass  
    
    def pretrain_weights(self, hidden_sizes, pretraining):
        if self.is_sparse and self.test.shape[0] > 1:    
            if self.gpu_data:
                data = sparse.csr_matrix(np.vstack([self.X.as_numpy_array(), self.test.as_numpy_array()]))
            else:        
                data = sparse.csr_matrix(sparse.vstack([self.X.todense(), self.test.todense()]))
        else:
            data = self.X
            
        rbm = RBM(data, hidden_sizes,pretraining, is_sparse = self.is_sparse, activation = self.activation, GPU = gpu.board_id_to_use, use_noise = False, logger = log)
        #sda = SDA(data, hidden_sizes, pretraining, is_sparse = self.is_sparse, activation = self.activation,
        #          GPU = gpu.board_id_to_use, logger=log, activation_gradient = self.activation_gradient)
        self.print_and_log('-------------------------------')
        self.print_and_log('BEGIN PRETRAINING')
        self.print_and_log('-------------------------------')
        return rbm.train()    
    
        
    def set_layer_sizes(self, hidden_sizes):   
        self.layer_sizes = []
        self.layer_sizes.append(self.X.shape[1])
        for layer in hidden_sizes:
            self.layer_sizes.append(layer)        
        
        if self.problem != 'regression':
            self.layer_sizes.append(np.int(np.max(self.y) + 1))
            log.info('recognized {0} as size of the output layer\n'.format(np.int(np.max(self.y) + 1)))
        else:
            self.layer_sizes.append(self.y.shape[1])        
              
        pass
        
    def create_weights(self, pretraining):
        self.w = []
        self.b = []
        self.m = []
        self.mb = []     
        
        in_layer = None
        out_layer = None
        for i, size in enumerate(self.layer_sizes):            
            if not in_layer:
                in_layer = size
            elif not out_layer:
                out_layer = size
            else:
                in_layer = out_layer
                out_layer = size
                
            #choose the function for initialization here, I got generally better results with sparse initialization
            init_func = self.util.create_sparse_weight         
            #init_func = self.util.create_uniform_rdm_weight
           
            if in_layer and out_layer:             
                if pretraining != None:
                    if len(pretraining)>= i:
                        self.w.append(gpu.garray(pretraining[i-1][0]))
                        self.b.append(gpu.garray(pretraining[i-1][1]))
                    else:        
                        self.w.append(gpu.garray(init_func(in_layer, out_layer)))
                        self.b.append(gpu.zeros((1, out_layer)))
                    self.m.append(gpu.zeros((in_layer, out_layer)))             
                    self.mb.append(gpu.zeros((1, out_layer)))  
                else:
                    self.w.append(gpu.garray(init_func(in_layer, out_layer)))
                    self.b.append(gpu.zeros((1, out_layer)))
                    self.m.append(gpu.zeros((in_layer, out_layer)))             
                    self.mb.append(gpu.zeros((1, out_layer)))  
            
        log.info('Weights:')            
        for weight in self.w:
            log.info(weight.shape)
                    
        log.info('')
        pass
    
    
    def create_rollback_point(self):        
        for i in range(len(self.w)):            
            np.save(self.path + 'rollback/w{0}'.format(i+1), self.w[i].as_numpy_array())
            np.save(self.path + 'rollback/b{0}'.format(i+1), self.b[i].as_numpy_array())
            np.save(self.path + 'rollback/m{0}'.format(i+1), self.m[i].as_numpy_array())
            np.save(self.path + 'rollback/mb{0}'.format(i+1), self.mb[i].as_numpy_array())
        pass
    
    
    def load_rollback_point(self):
        weights = []
        biases = []
        momentum = []
        momentum_bias = []
        while(1):
            i = 1
            if os.path.exists(self.path + 'rollback/w{0}'.format(i)):
                weights.append(np.load(self.path + 'rollback/w{0}'.format(i)))
                biases.append(np.load(self.path + 'rollback/b{0}'.format(i)))
                momentum.append(np.load(self.path + 'rollback/m{0}'.format(i)))
                momentum_bias.append(np.load(self.path + 'rollback/mb{0}'.format(i)))
                i+=1
            else:
                break
        self.w = weights
        self.b = biases
        self.m = momentum
            
        
    def train(self):        
        for epoch in range(self.epochs):
            self.begin_of_epoch(epoch)
            for set_name in self.set_names:
                for start_idx in range(0,self.X.shape[0],self.batch_size):             
                    if self.allocate_batch(start_idx, set_name):
                        break                
                    
                    if set_name == 'train' and self.momentum_type == 1:
                        self.nesterov_momentum()
                    self.feedforward(epoch, set_name)      
                    if set_name == 'train':
                        self.backprop()
                        self.gradient_updates()     
                
            self.end_of_epoch(epoch)
            self.update_parameters(epoch)  
            
            if self.stop_at_score != None:
                if (self.last_train_error <= self.stop_at_score and self.best_result_type == 'decreasing') or (self.last_train_error >= self.stop_at_score and self.best_result_type == 'increasing'):
                    self.print_and_log('Final training error reached: {0}'.format(self.stop_at_score))
                    self.print_and_log('Stopping training...')
                    break
                
                      
        self.print_and_log('Best error: {0}'.format(self.best_result))
        if self.set_sizes[0] == 1 or  self.stop_at_score != None:
            self.print_and_log('Creating end of training predictions...')
            self.predict()           
            if self.save_weights:
                self.create_rollback_point()     
                
                       
    
    def begin_of_epoch(self, epoch):
        for set_name in self.set_names:
            if set_name != 'train':
                self.set_error_by_epoch[set_name].append(0)
        self.results.clear()
        self.time_values.clear()
                
            
    def allocate_batch(self, start_idx, set_name = 'train'): 
        self.timer_logger('allocate_batch', time.time())  
              
        if set_name == 'train' or set_name == 'train_error' or set_name == 'no_label_test':             
            end_idx = (start_idx)+self.batch_size        
        elif set_name == 'cv': 
            start_idx += self.set_boundaries['train']           
            end_idx = start_idx + self.batch_size            
        elif set_name == 'test':
            start_idx += self.set_boundaries['cv']                        
            end_idx = start_idx + self.batch_size       
                      
        if set_name == 'no_label_test':          
            end_idx = self.test.shape[0] if end_idx >= self.test.shape[0] else end_idx
            if self.gpu_data:
                self.batch = self.test[start_idx:end_idx,:]
            else:
                self.batch = gpu.garray(self.test[start_idx:end_idx,:] if not self.is_sparse else self.test[start_idx:end_idx,:].todense())             
            if start_idx >= end_idx: 
                self.timer_logger('allocate_batch', time.time())   
                return True    
        else:          
            end_idx = self.set_boundaries[set_name] if end_idx >= self.set_boundaries[set_name] else end_idx            
            if start_idx >= end_idx: 
                self.timer_logger('allocate_batch', time.time())   
                return True
            
            if self.gpu_data:
                self.batch = self.X[start_idx:end_idx,:]
            else:
                self.batch = gpu.garray(self.X[start_idx:end_idx,:] if not self.is_sparse else self.X[start_idx:end_idx,:].todense())
                        
            self.batch_y = self.y[start_idx:end_idx]
                
            if self.problem == 'regression':
                self.batch_y = gpu.garray(self.batch_y)
            
            self.current_batch_size = (end_idx-start_idx)
        self.timer_logger('allocate_batch', time.time())        
        return start_idx >= end_idx
    
    def nesterov_momentum(self):
        self.timer_logger('nesterov_momentum', time.time())          
        for i in range(len(self.w)):
            if 'grads' in self.results:
                self.w[i] += (self.m[i]*self.momentum)
                self.b[i] += (self.mb[i]*self.momentum)
        self.timer_logger('nesterov_momentum', time.time())           
    
    def input_to_hidden(self, set_name = 'train'):
        self.timer_logger('input_to_hidden {0}'.format(type), time.time()) 
        self.results['activations'] = []     
        if set_name == 'train':            
            self.results['activations'].append([self.batch, self.w[0], self.b[0]])   
            dropped_out = self.batch * (gpu.rand(self.current_batch_size,self.X.shape[1]) > self.dropout[0]) 
            self.results['current']  = gpu.dot(dropped_out,self.w[0])+self.b[0]
        else:                               
            self.results['current'] = gpu.dot(self.batch,self.w[0]) + self.b[0]
        self.timer_logger('input_to_hidden {0}'.format(type), time.time()) 
        
    def hidden_to_output(self, set_name = 'train'):   
        self.timer_logger('hidden_to_output {0}'.format(type), time.time()) 
        i = 0   
        for weight, bias in zip(self.w, self.b):
            if i > 0: #ignore the first weight that goes from inputs to first hidden layer
                if set_name == 'train':                            
                    self.results['activations'].insert(0, [self.activation(self.results['current'])   , weight])            
                    self.results['current'] = gpu.dot(self.results['activations'][0][0] * 
                                                  (gpu.rand(self.results['activations'][0][0].shape[0],self.results['activations'][0][0].shape[1]) > self.dropout[1]), #dropout
                                                   weight) + bias                    
                else:
                    self.results['current'] =  gpu.dot(self.activation(self.results['current'])* (1 - self.dropout[1]), weight) + bias
          
            i += 1
        self.timer_logger('hidden_to_output {0}'.format(type), time.time()) 
                    
    def output_and_cost(self, epoch, set_name = 'train'):       
        self.timer_logger('output_and_cost {0}'.format(set_name), time.time())  
   
        self.results['current'] = self.output(self.results['current'])        
        if self.problem == 'regression' and self.clip_values == 1:
            # clip values into the [0,1] range
            self.results['current']  = (self.results['current'] *(self.results['current'] >= 0)) 
            self.results['current'] = (((self.results['current'] < 1)*self.results['current']) + (self.results['current'] > 1))        
        
        if set_name != 'train':       
            if set_name == 'no_label_test':                  
                if 'prediction_test' not in self.results:
                    if self.problem == 'classification':
                        self.results['prediction_test'] = np.argmax(self.results['current'].as_numpy_array(),axis=1)
                    else:                       
                        self.results['prediction_test'] = self.results['current'].as_numpy_array()    
                else:          
                    if self.problem == 'classification':          
                        self.results['prediction_test'] = np.hstack([self.results['prediction_test'],np.argmax(self.results['current'].as_numpy_array(),axis=1)])  
                    else:                      
                        self.results['prediction_test'] = np.vstack([self.results['prediction_test'],self.results['current'].as_numpy_array()])  
            elif set_name == 'cv_predict':
                if self.create_cv_predictions and set_name == 'cv_predict':
                    if 'prediction_cv'not in self.results:                            
                        self.results['prediction_cv'] = self.results['current'].as_numpy_array()
                    else:
                        self.results['prediction_cv'] = np.vstack([self.results['prediction_cv'],self.results['current'].as_numpy_array()])              
            else:                  
                if self.problem == 'classification':   
                    self.set_error_by_epoch[set_name][epoch] += (np.sum(np.equal(np.argmax(self.results['current'].as_numpy_array(),axis=1),self.batch_y.T)))
                else: 
                    self.set_error_by_epoch[set_name][epoch] += gpu.sqrt(gpu.sum(((self.results['current']-self.batch_y)**2)*float(self.batch.shape[0]))/float(self.y.shape[1]))                     
                    
                if self.cost == 'auc':
                    if self.problem == 'regression':
                        if set_name + ' roc_auc' not in self.results:
                            self.results[set_name + ' roc_auc'] = ([np.matrix(self.results['current'].as_numpy_array()).T, np.matrix(self.batch_y).T]) 
                        else:
                            self.results[set_name + ' roc_auc'] = [np.hstack([self.results[set_name + ' roc_auc'][0],np.matrix(self.results['current'].as_numpy_array()).T]),
                                                               np.hstack([self.results[set_name + ' roc_auc'][1],np.matrix(self.batch_y).T])]
                
                    else:          
                        if set_name + ' roc_auc' not in self.results:
                            self.results[set_name + ' roc_auc'] = ([np.matrix(self.results['current'].as_numpy_array()[:,1]).T, np.matrix(self.batch_y)])                             
                        else:
                            self.results[set_name + ' roc_auc'] = [np.vstack([np.matrix(self.results[set_name + ' roc_auc'][0]),np.matrix(self.results['current'].as_numpy_array()[:,1]).T]),
                                                               np.vstack([self.results[set_name + ' roc_auc'][1],np.matrix(self.batch_y)])]
                    
                    
        self.timer_logger('output_and_cost {0}'.format(set_name), time.time())  

            
            
    def feedforward(self, epoch, set_name = 'train'):        
        self.input_to_hidden(set_name)
        self.hidden_to_output(set_name)
        self.output_and_cost(epoch, set_name)       
        
        
    def backprop(self):            
        self.timer_logger('backprop', time.time())   
        self.results['grads'] = []
        self.results['bias_grads'] = []   
        if self.problem == 'classification':   
            #assumes softmax + cross entropy so that both gradients cancel out to give: error = y-t   
            self.results['error'] = self.results['current'] - gpu.garray(self.util.create_t_dataset(self.batch_y))     
        else:
            #assumes linear unit + squared error cost function so that both gradients cancel out to give: error = y-t  
            self.results['error'] = (self.results['current'] - gpu.garray(self.batch_y)) 
            
        for pair in self.results['activations']:
            activation = pair[0]
            weight = pair[1] 
            
            gradient = self.activation_gradient(activation)             
            self.results['grads'].insert(0,gpu.dot(activation.T,self.results['error']))     
            self.results['bias_grads'].insert(0,gpu.dot(gpu.ones((1,self.results['error'].shape[0])),self.results['error']))         
            self.results['error'] = gpu.dot(self.results['error'],weight.T)*gradient
            
        self.timer_logger('backprop', time.time())   
                
                
    def gradient_updates(self):
        self.timer_logger('gradient_updates', time.time())   
        current_batch_size = 1.0/(self.results['activations'][0][0].shape[0]*1.0)
        multiplier = self.learning_rate*current_batch_size
        for i in range(len(self.w)):
            if self.momentum_type == 0:
                #no momentum
                self.w[i] = self.w[i] - (self.results['grads'][i]*multiplier)
                self.b[i] = self.b[i] - (self.results['bias_grads'][i]*multiplier)
            elif self.momentum_type == 1:
                # Nesterov's accelerated gradient      
                if self.L1 > 0 or self.L2 > 0:
                    if self.L1 > 0 and self.L2 > 0:
                        self.results['grads'][i] = (self.results['grads'][i] + (self.w[i]*self.L2) + ((self.w[i] != 0))*self.L1)*multiplier
                        self.results['bias_grads'][i] = (self.results['bias_grads'][i] + (self.b[i]*self.L2) + ((self.b[i] != 0))*self.L1)*multiplier    
                    elif self.L1 > 0:
                        self.results['grads'][i] = (self.results['grads'][i] + ((self.w[i] != 0))*self.L1)*multiplier
                        self.results['bias_grads'][i] = (self.results['bias_grads'][i] + ((self.b[i] != 0))*self.L1)*multiplier    
                        
                    else: 
                        self.results['grads'][i] = (self.results['grads'][i] + (self.w[i]*self.L2))*multiplier
                        self.results['bias_grads'][i] = (self.results['bias_grads'][i] + (self.b[i]*self.L2))*multiplier    
                else:
                    self.results['grads'][i] = self.results['grads'][i]*multiplier
                    self.results['bias_grads'][i] = self.results['bias_grads'][i]*multiplier
                    
                #subtract Nesterov's accelerated gradient
                self.w[i] -= (self.m[i]*self.momentum)
                self.b[i] -= (self.mb[i]*self.momentum)
                #update weights and the momentum vector
                self.m[i] = (self.momentum*self.m[i]) - self.results['grads'][i]
                self.mb[i] = (self.momentum*self.mb[i]) - self.results['bias_grads'][i]              
                self.w[i] +=  (self.m[i])
                self.b[i] += (self.mb[i])
                
        self.timer_logger('gradient_updates', time.time())   
              
        
    def end_of_epoch(self, epoch):
        '''prints errors and handles early stopping
        '''
        self.timer_logger('end_of_epoch', time.time())   
        self.print_and_log('EPOCH {0}'.format(epoch+1)) 
        for set_name in self.set_names:     
            if set_name != 'train':
                if self.set_sizes_absolute[set_name] > 0:
                    if self.problem == 'classification':
                        self.set_error_by_epoch[set_name][epoch] = 1.0 - (self.set_error_by_epoch[set_name][epoch]/ (self.set_sizes_absolute[set_name]*1.0))
                    else:                       
                        self.set_error_by_epoch[set_name][epoch] = (self.set_error_by_epoch[set_name][epoch]/ (self.set_sizes_absolute[set_name]*1.0))
                             
                if self.set_error_by_epoch[set_name][epoch] != 0:
                    self.print_and_log('{1} {2} error: {0}'.format(self.set_error_by_epoch[set_name][epoch], set_name, 'misclassification' if self.problem == 'classification' else 'root mean squared' ))
                
                auc = 0
                if set_name + ' roc_auc' in self.results: 
                    if self.problem == 'classification':
                        fpr, tpr, thresholds = metrics.roc_curve(self.results[type + ' roc_auc'][1], self.results[type + ' roc_auc'][0], pos_label=1)
                    else:
                        fpr, tpr, thresholds = metrics.roc_curve(self.results[type + ' roc_auc'][1].T, self.results[type + ' roc_auc'][0].T, pos_label=1)
                    auc = metrics.auc(fpr,tpr) 
                    self.print_and_log(type + ' auc: {0}'.format(auc) )
                                 
                if self.cost == 'auc':
                    error = auc    
                else:
                    error = self.set_error_by_epoch['cv'][epoch]
                    self.last_train_error = error     
                    
               
                self.last_train_error = self.set_error_by_epoch['train_error'][epoch]                              
                     
                self.timer_logger('end_of_epoch', time.time())         
                if ((error > self.best_result[1] and self.best_result_type == 'increasing') or  
                   (error < self.best_result[1] and self.best_result_type == 'decreasing')) and set_name == 'cv':
                    
                    if epoch > 0:
                        self.create_prediction = True if self.create_prediction == True or self.best_result[0]+1 != epoch else False                       
                    self.best_result[1] = error         
                    self.best_result[0] = epoch                 
                                        
                    s = 'New best error: {0}'.format(error)                      
                    self.print_and_log(s)
                    if self.test.shape[0] > 1 and self.create_prediction:
                        self.print_and_log('creating predictions...')
                        self.predict()
                        if self.save_weights:
                            self.create_rollback_point()
        if self.time_printing:
            self.print_timelog() 
                
                
    def predict(self):   
        self.timer_logger('predict', time.time())   
        if 'prediction_test' in self.results: self.results.pop('prediction_test')  
        if 'prediction_cv' in self.results: self.results.pop('prediction_cv')  
        #create predictions for test set
        for start_idx in range(0,self.test.shape[0], self.batch_size):
            self.allocate_batch(start_idx, set_name='no_label_test')            
            self.feedforward(0, set_name='no_label_test')
          
        #create predictions for the CV set
        for start_idx in range(0,self.set_sizes_absolute['cv'], self.batch_size):
            self.allocate_batch(start_idx, set_name='cv')
            self.feedforward(0, set_name='cv_predict')
               
        prediction_test = self.results['prediction_test'] 
        #stack prediction values with the IDs for the csv file
        if self.ids == None and self.problem == 'classification':
            prediction = np.vstack([np.arange(0,self.test.shape[0]), self.results['prediction_test']]).T
        elif self.ids != None and self.problem == 'classification':
            prediction = np.hstack([np.matrix(self.ids), np.matrix(self.results['prediction_test']).T])
        elif self.ids != None and self.problem == 'regression':
            prediction = np.hstack([self.ids, self.results['prediction_test']])
        elif self.problem == 'regression':
            prediction = np.hstack([np.arange(0,self.test.shape[0]).reshape(self.test.shape[0],1), self.results['prediction_test']])  
                  
        prediction_cv = self.results['prediction_cv']
        if self.problem == 'regression':
            col_format = '%i,' + '%f,'*(self.w[-1].shape[1]-1) + '%f'#
        else:
            col_format = '%i,' + '%f'
        if self.problem == 'classification':            
            np.savetxt ('C:/net/results/predict_GPU{0}.csv'.format(gpu.board_id_to_use), prediction,col_format, delimiter=',')              
            np.save('C:/net/results/prediction_{1}GPU{0}.npy'.format(gpu.board_id_to_use,self.file_id ),prediction_test)
            np.save('C:/net/results/prediction_cv_{1}GPU{0}.npy'.format(gpu.board_id_to_use,self.file_id ),prediction_cv)  
            
        else:       
            np.savetxt ('C:/net/results/predict_GPU{0}.csv'.format(gpu.board_id_to_use), prediction,col_format, delimiter=',')  
            np.save('C:/net/results/prediction_{1}GPU{0}.npy'.format(gpu.board_id_to_use,self.file_id ),prediction_test) 
            np.save('C:/net/results/prediction_cv_{1}GPU{0}.npy'.format(gpu.board_id_to_use,self.file_id ),prediction_cv)  
        
        self.timer_logger('predict', time.time())          
            
    def update_parameters(self, epoch):
        self.momentum += 0.001          
        if self.momentum > self.end_momentum: 
            self.momentum = self.end_momentum       
          
        #reduce the learning rate linearly and half dropout at transition   
        if epoch > self.transition:
            self.momentum_type = 0
            self.learning_rate  = self.initial_learning_rate / float((epoch - self.transition))  
            if (epoch-1 - self.transition) % 100 == 0:
                self.print_and_log('setting dropout to {0} and {1}'.format(self.dropout[0]/2.0, self.dropout[1]/2.0))  
                self.dropout[0] /=2.0
                self.dropout[1] /=2.0
            
     
'''
X = np.load('C:/data/MNIST_train.npy')
test = np.load('C:/data/MNIST_test.npy')
u = Util()
X =  np.vstack([X, test])
y = X[:,0]
X = X[:,1:]

net = Deep_net([800, 1200, 1600, 2400], data = [X, y, None], set_sizes = [0.8571429,(1-0.8571429),0], pretraining = [15,10,7,5], printing = False)
'''

#net = Deep_net(None, [800], data = [X, u.create_t_matrix(y), X], set_sizes = [0.8571429,(1-0.8571429),0], problem='regression',output_dim = 10, learning_rate = 0.1)


'''
#net = Deep_net(None, [800, 1200], pretraining = [10,10], data = [X, y, None], set_sizes = [0.8571429,0,(1-0.8571429)])
#net = Deep_net(None, [800, 1200],pretraining = [10,10], data = [X, y, None], set_sizes = [0.6571429,0.2,(1-0.8571429)
'''