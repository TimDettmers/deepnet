import numpy as np
import matplotlib.pyplot as plt
import sys
import select
import h5py
from scipy import sparse
import time

class Util:
    
    
    def softmax(self, X):
        '''numerically stable softmax function
        '''
        max_row_values = np.matrix(np.max(X,axis=1)).T
        result = np.exp(X - max_row_values)
        sums = np.matrix(np.sum(result,axis=1))        
        return result/sums
        
    
    def evolutionary_optimization(self, weight, X,y, func, percentile=3, population= 500, noise_variance=0.5, direction='max', epochs = 12):        
        best_error = 0 if direction == 'max' else 1
        rdm = np.random.RandomState(1234)
        best_noise = 0
        t0 = time.time()
        best_weight = weight
        for epoch in range(epochs):
            best_weights = []
            best_errors = []
            best_noises = []
            for i in range(population):
                noise = rdm.normal(0,noise_variance,(weight.shape))
                #noise = gpu.randn(m,1,1)/5.0
                current = func(X,y, noise, weight)
                #print 'Cross validation error: {0}'.format(current)
                best_weights.append(weight)
                best_errors.append(current)
                best_noises.append(noise)
                if direction == 'max' and current > best_error or (direction == 'min' and current < best_error):                
                    best_error = current            
                    best_noise = noise  
                    best_weight = weight  
        
            print 'EPOCH: {0}, best_error: {1}'.format(epoch,best_error)  
            if direction == 'max':
                idx = np.where(np.array(best_errors) >= np.percentile(best_errors, q=percentile))[0]
            else:
                idx = np.where(np.array(best_errors) <= np.percentile(best_errors, q=percentile))[0]
                
            weight = np.mean((np.array(best_weights)[idx] + np.array(best_noises)[idx]),axis=0)            
         
            
        print best_error
        print best_noise.T 
        print best_weight.T
        print time.time() - t0
    
    def strings_to_classes(self, strings):
        ret_classes = []
        dict_classes = {}
        i = 0
        for val in strings:
            if val not in dict_classes.keys():
                dict_classes[val] = i            
                ret_classes.append(i)
                i+=1
            else:
                ret_classes.append(dict_classes[val]) 
                
        return np.array(ret_classes)
    
    def hyperparameter_fitting(self, fun, data, means, lower_vals, upper_vals, positive=True, iter=20): 
        def get_new_params(data, means, lower_vals, upper_vals, positive=True):
            data = np.array(data)  
            ret_params = np.zeros_like(np.array(means))
            if data.shape[0] > 5:
                best_result_idx = np.argmax(data[:,-1])
                means = data[best_result_idx,:-1]        
                
            for i, mean in enumerate(means):
                    lower  = lower_vals[i]
                    upper = upper_vals[i]  
                    if data.shape[0] > 10:   
                        for j in range(len(means)):
                            upper = np.percentile(data[:,-1], 75)                          
                            variance = np.var(data[data[:,-1] > upper,j],axis=0)  
                            #mean = np.mean(data[data[:,-1] > upper,j],axis=0)
                    else:     
                        variance = ((upper - lower)/ (2* 1.96))**2
                    
                    rdm_value = np.random.normal(mean,variance)
                    if positive:                
                        while rdm_value <= 0:
                            rdm_value = np.random.normal(mean,variance)                
                        
                    ret_params[i] = rdm_value     
                
            return ret_params
        
        params = get_new_params(data, means, lower_vals, upper_vals)
        param_data = []
        for epoch in range(iter):
            cv_score = fun(params)
            print 'CV score: {0}'.format(cv_score)
            param_data.append(params.tolist() + [cv_score])
            params = get_new_params(param_data,means, lower_vals, upper_vals)
    
        print 'Best parameter: {0}'.format(get_new_params(param_data,means, lower_vals, upper_vals))
    
    def create_t_matrix(self, y):
        classes = np.max(y)
        t = np.zeros((y.shape[0], classes+1))
        for i in range(y.shape[0]):
            t[i, y[i]] = 1
            
        return t
    
    def create_balanced_set_index(self, y, X):   
        labels_and_cases = []
        labels = np.max(y)
        a = np.zeros((labels+1,))
        for i in range(a.shape[0]):
            a[i] = np.sum(y==i)       
            labels_and_cases.append(np.where(y==i)[0].tolist())
             
        a_original = a.copy()   
        X_new = np.zeros((X.shape))   
        y_new = np.zeros((X.shape[0]))
        for row in range(X.shape[0]):
            next_label = np.argmax(a)
            if len(labels_and_cases[next_label]) > 0:  
                y_new[row] = next_label
                X_new[row] = X[labels_and_cases[next_label].pop()]
            a += a_original*(np.arange(0,labels+1)!=next_label)
        
        return y_new, X_new
    
    def create_balanced_index_vector(self, y):   
        labels_and_cases = []
        labels = np.max(y)
        a = np.zeros((labels+1,))
        for i in range(a.shape[0]):
            a[i] = np.sum(y==i)       
            labels_and_cases.append(np.where(y==i)[0].tolist())
             
        a_original = a.copy()          
        y_idx = []
        for row in range(y.shape[0]):
            next_label = np.argmax(a)
            if len(labels_and_cases[next_label]) > 0:  
                y_idx.append(labels_and_cases[next_label].pop())              
            a += a_original*(np.arange(0,labels+1)!=next_label)
        
        return np.array(y_idx)
    
    def save_sparse_matrix(self, filename,x):    
        x = sparse.csr_matrix(x)
        data=x.data
        indices=x.indices
        indptr=x.indptr
        shape=x.shape
        file = h5py.File(filename,'w')
        file.create_dataset("indices", data=indices)
        file.create_dataset("indptr", data=indptr)
        file.create_dataset("data", data=data)
        file.create_dataset("shape", data=shape)
        file.close()

    def load_sparse_matrix(self, filename):
        f = h5py.File(filename,'r')
        z = sparse.csr_matrix( (f['data'],f['indices'],f['indptr']), shape=f['shape'])
        return z
    
    def create_batches(self, X, size):
        count = np.round(X.shape[0]/(1.0*size),0)
        return np.array(np.split(X,count))
    
    def create_sparse_weight(self, input_size, output_size, sparsity = 15):   
        rdm = np.random.RandomState(1234)     
        weight = np.zeros((input_size, output_size))
        for axon in range(output_size):            
            idxes = rdm.randint(0,input_size, (sparsity,))
            rdm_weights = rdm.randn(sparsity)
            for idx, rdm_weights in zip(idxes, rdm_weights):
                weight[idx,axon] = rdm_weights                
        return weight
    
    def create_uniform_rdm_weight(self,input_size,output_size):
        rdm = np.random.RandomState(1234)        
        return rdm.uniform(low=-4*np.sqrt(6./(input_size+output_size)),
                        high=4*np.sqrt(6./(input_size+output_size)),
                        size=(input_size,output_size))
    
       
    def create_t_dataset(self, y):        
        if y != None:
            Y = np.matrix(y)
            Y = Y.T if Y.shape[0] == 1 else Y
            
            no_labels = np.max(y)
            t = np.zeros((Y.shape[0],no_labels+1))
            for i in range(Y.shape[0]):
                t[i,Y[i,0]] = 1
                
            return t
        else:
            return None   
        
    def shuffle_set(self, data_set_X, data_set_y, data_set_t):
        n = data_set_X.shape[0]
        rdm_idx = np.arange(0,n)
        np.random.shuffle(rdm_idx)
        new_X = np.zeros((data_set_X.shape))
        new_y = np.zeros((data_set_y.shape))
        new_t = np.zeros((data_set_t.shape))
        for i in range(n):
            new_X[i,:] = data_set_X[rdm_idx[i],:]
            new_y[i] = data_set_y[rdm_idx[i]]
            new_t[i,:] = data_set_t[rdm_idx[i],:]
    
    
    def plot_results(self, valid, train, epochs, filename):
        plt.hold(True) 
        print 'Printing result...'
        plt.axis([0,epochs,0,0.05])
        plt.title('Epochs: ' + str(epochs) + ', '  +'Hidden layer units: ')
        plt.plot(range(epochs),valid,color='blue')
        plt.plot(range(epochs),train,color='red')
        plt.tight_layout()
        plt.savefig(filename +'.png')
        plt.hold(False)
     
    def plot_weights(self, weight, filename):
        print 'Printing weights...'
        hist, bins = np.histogram(weight,bins = 50)
        width = 0.7*(bins[1]-bins[0])
        center = (bins[:-1]+bins[1:])/2
        plt.bar(center, hist, align = 'center', width = width)
        plt.savefig(filename + '.png')
        
    def heardEnter(self):
        i,o,e = select.select([sys.stdin],[],[],0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False    