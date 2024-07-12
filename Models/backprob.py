# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 05:40:11 2022

@author: Hussein Aly
"""
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from math import sqrt
from scipy.special import expit
from sklearn import metrics
from easydict import EasyDict
import torch
import torch.nn as nn
from scipy import stats
import torch.nn.functional as functional
from torch.optim import SGD, Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.utils.class_weight import compute_class_weight
import datetime
import os 
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import RepeatedStratifiedKFold


# class EarlyStopper:
#     def __init__(self, patience=1, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = np.inf

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

class EarlyStopping:
    def __init__(self, patience=10, smoothing=10, delta=0, checkpoint_name='checkpoint.pt'):
        self.patience = patience
        self.smoothing = smoothing
        self.delta = delta
        self.checkpoint_name = checkpoint_name
        self.best_loss = np.Inf
        self.early_stop = False
        self.losses = []
        self.ema = None
        self.counter = 0

    def __call__(self, val_loss, model):
        if len(self.losses) < self.smoothing:
            self.ema = val_loss
        else:
            self.ema = moving_average(self.losses[-self.smoothing:] , self.smoothing)
        
        if self.ema < self.best_loss - self.delta:
            self.best_loss = self.ema
            self.save_checkpoint(self.ema, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.best_epoch = len(self.losses) - self.patience

        self.losses.append(val_loss)
        
    def save_checkpoint(self, val_loss, model):
        torch.save({'model_state_dict': model.state_dict(), 
                    'val_loss': val_loss}, self.checkpoint_name)

    def load_checkpoint(self, model):
        checkpoint = torch.load(self.checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint['val_loss']
        print(f"Loaded checkpoint with validation loss: {self.best_loss}")
        # remove checkpoint file
        os.remove(self.checkpoint_name)
        print(f"Checkpoint File {self.checkpoint_name} Removed")

class Perceptron(nn.Module):
    def __init__(self, n_layers , nodes , n_features , n_classes , p_drop, device , direct_link) :
        super(Perceptron, self).__init__()
        self.nodes = nodes
        self.n_features = n_features
        self.n_classes = n_classes
        self.d = device
        self.direct_link = direct_link # the direct link between the input and the output layer
        # init layers
        self.layers =  nn.ModuleList()
        # # append input layer
        # self.layers.append(nn.Linear(n_features , nodes))
        # append the hidden layers
        for layer_id in range(n_layers) :
            seq_layer =  nn.Sequential(
            nn.Linear(n_features if layer_id ==0 else nodes , nodes),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.BatchNorm1d(self.nodes, affine = True, momentum=0.1, track_running_stats=True),)
            self.layers.append(seq_layer)

        # append the output layer
        if direct_link :
            self.layers.append(nn.Linear(nodes + n_features, n_classes))
        else :
            self.layers.append(nn.Linear(nodes, n_classes))

    def forward(self , X):
        X_raw = X.clone()
        for layer in self.layers[:-1] :
            X = layer(X)
        grouped = torch.cat([X_raw , X] , axis=1 )
        if self.direct_link :
            out = nn.functional.softmax(self.layers[-1](grouped), 1)
        else :
            out = nn.functional.softmax(self.layers[-1](X), 1)
        return out

class SNNPerceptron(nn.Module):

    def __init__(self, n_layers ,  nodes, n_features, n_classes , p_drop, device ,direct_link ):
        super(SNNPerceptron, self).__init__()
        self.nodes = nodes
        self.n_features = n_features
        self.n_classes = n_classes
        self.d = device
        self.direct_link = direct_link # the direct link between the input and the output layer
        # init layers
        self.layers =  nn.ModuleList()
        # append input layer
        # self.layers.append(nn.Linear(n_features , nodes))

        for layer_id in range(n_layers) :
            seq_layer = nn.Sequential(
            nn.Linear(n_features if layer_id ==0 else nodes , nodes),
            nn.SELU(),
            nn.AlphaDropout(p=p_drop),)
            self.layers.append(seq_layer)

        # append the output layer
        if direct_link :
            self.layers.append(nn.Linear(nodes + n_features, n_classes))
        else :
            self.layers.append(nn.Linear(nodes, n_classes))

        for param in self.layers.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
    

    def forward(self , X):
        X_raw = X.clone()
        for layer in self.layers[:-1] :
            X = layer(X)
        grouped = torch.cat([X_raw , X] , axis=1 )
        if self.direct_link :
            out = nn.functional.softmax(self.layers[-1](grouped), 1)
        else :
            out = nn.functional.softmax(self.layers[-1](X), 1)
        return out

class MLP_model():
    def __init__(self, classes, nodes , layers= 5 , epochs = 10, learning_rate = 1e-3,
                 device='cpu', batch_percentage= 0.05, weight_decay = 0, l1_weight= 0,
                output_loc = None ,direct_link = True, seed = -1 , verbose = 1 , p_drop = 0, snn= False):
        self.classes = classes
        self.nodes = nodes
        self.layers = layers
        self.epochs = epochs
        self.learning_rate  = learning_rate
        self.d = device
        self.batch_percentage = batch_percentage
        self.weight_decay = weight_decay
        self.l1_weight = l1_weight
        self.direct_link = direct_link
        self.seed = seed
        self.verbose = verbose 
        self.output_loc = output_loc
        self.snn = snn
        self.p_drop = p_drop

    def train(self, X , y, X_val = None , y_val = None, epochs = None ):

        run_id =  datetime.datetime.now().strftime("%d_%m_%YT_%H_%M_%S")
        if self.seed != -1 :
            self.fix_seed(self.seed)
        
        if epochs is None :
            epochs = self.epochs

        X = torch.Tensor(X).float().to(self.d)
        y = torch.Tensor(y).float().to(self.d)
        if not X_val is None : 
            X_val = torch.Tensor(X_val).float().to(self.d)
            y_val = torch.Tensor(y_val).float().to(self.d)

        if self.snn : 
            model = SNNPerceptron(self.layers, self.nodes, X.shape[1], y.shape[1],p_drop=self.p_drop , device = self.d,
            direct_link = self.direct_link )
        else :
            model = Perceptron(self.layers, self.nodes, X.shape[1], y.shape[1], p_drop=self.p_drop, device = self.d,
                        direct_link = self.direct_link )
        model.to(self.d)

        loss_fn = nn.CrossEntropyLoss()

        opt = torch.optim.Adam(model.parameters() , lr = self.learning_rate, weight_decay=self.weight_decay)
        
        # batch steps
        n_steps = np.ceil(1 / self.batch_percentage)

        train_loss_lst = []
        val_loss_lst = []
        val_acc_lst = []
        train_acc_lst = []


        train_loss , train_acc = self.eval(model , loss_fn, X, y)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        self.__consol__(f"Epoch {0} -> train_loss = {train_loss:0.2f}, trian_acc = {train_acc * 100:0.2f}" ,level=1)

        if not X_val is None :
            val_loss , val_acc = self.eval(model , loss_fn, X_val, y_val)
            val_loss_lst.append(val_loss)
            val_acc_lst.append(val_acc)
            self.__consol__(f"Epoch {0} -> val_loss = {val_loss:0.2f}, val_acc = {val_acc * 100:0.2f}", level=1)
        
        
        
        early_stopper = EarlyStopping(patience=10, smoothing=10 , checkpoint_name=f"{self.output_loc}/MLP_{run_id}.pt", delta=0)

        print(f"starting Model training")
        for epoch in range(int(epochs)) :

            self.train_loop(X, y, model , loss_fn , opt)

            train_loss , train_acc = self.eval(model , loss_fn, X, y)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)
            self.__consol__(f"Epoch {epoch} -> train_loss = {train_loss:0.2f}, trian_acc = {train_acc * 100:0.2f}" ,level=1)

            if not X_val is None :
                val_loss , val_acc = self.eval(model , loss_fn, X_val, y_val)
                val_loss_lst.append(val_loss)
                val_acc_lst.append(val_acc)
                self.__consol__(f"Epoch {epoch} -> val_loss = {val_loss:0.2f}, val_acc = {val_acc * 100:0.2f}", level=1)
                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    self.__consol__(f"Early stopping at epoch {epoch}", level=1)
                    break           

        if not X_val is None :
            early_stopper.load_checkpoint(model)

        self.model = model
        
        train_loss_lst = np.array(train_loss_lst)
        train_acc_lst = np.array(train_acc_lst)
        
        if early_stopper.early_stop : 
            best_epoch = early_stopper.best_epoch
        else : 
            best_epoch= epochs

        
        if not X_val is None :
            val_loss_lst = np.array(val_loss_lst)
            val_acc_lst = np.array(val_acc_lst)
            return  best_epoch
        else :
            return None 
        
    def fix_seed(self , seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train_loop(self , X , y, model , loss_fn , opt) :

        model.train()

        batch_size =  int(np.ceil(self.batch_percentage * X.shape[0]))
        indexs = torch.randperm(X.shape[0]) # premutated to have random shuffling

        for batch_id in range(0 , X.shape[0] , batch_size) :
            batch_idx = indexs[batch_id : batch_id + batch_size ]
            train_pred = model(X[batch_idx])
            loss = loss_fn(train_pred , y[batch_idx])

            # compute L1 loss
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.reshape(-1))
            l1 = self.l1_weight * torch.abs(torch.cat(l1_parameters)).sum()
            loss += l1

            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, X):
        self.model.eval()
        X = torch.Tensor(X).float().to(self.d)
        pred_score = self.model(X)
        return pred_score.detach().cpu().numpy()

    def eval(self , model , loss_fn , X , y):
        model.eval()
        with torch.no_grad():
            train_pred = model(X)
            loss = loss_fn(train_pred , y)
            loss = loss.detach().cpu().numpy()

            train_pred = train_pred.cpu().detach().numpy()
            train_acc = accuracy_score(y.argmax(1).cpu() , train_pred.argmax(1))

        return loss , train_acc

    def __consol__(self, string : str , level : int ) : 
        if level <= self.verbose : 
            print(string)

class EdPreceptron(nn.Module):
    def __init__(self, features, raw_features, classes , nodes, device, gamma, beta, bn_learnable, track_running_stats , p_drop , ODL):
        super(EdPreceptron, self).__init__()

        # init params
        self.features = features
        self.raw_features = raw_features
        self.classes = classes
        self.nodes = nodes
        self.d = device
        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats,
        self.p_drop = p_drop
        self.ODL = ODL # output direct link 
        # init layers
        self.layer = nn.Sequential(
            nn.Linear(self.features , self.nodes),
            nn.ReLU(),
            nn.Dropout(p=p_drop),
            nn.BatchNorm1d(self.nodes, affine = True, momentum=0.1, track_running_stats=self.track_running_stats),
            
        )
        self.layer.apply(self.__init_weights__)

        self.output = nn.Sequential(
                nn.Linear(self.nodes + self.raw_features if ODL else self.nodes , classes),
                nn.Softmax(dim=1))

    def __init_weights__(self, m):
        if isinstance(m, nn.Linear):
            # y = m.in_features
            # m.weight.data.normal_(0.0,1/np.sqrt(y))
            # m.bias.data.fill_(0)
            m.weight.data.uniform_(-1 , 1)
            m.bias.data.uniform_(0, 1)
        # if bn is not learnable, set the values for gama and beta manually and freez their leanring
        elif isinstance(m, nn.BatchNorm1d) and (m.weight is not None) and (m.bias is not None) :
            if  self.bn_learnable : 
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)
                # m.weight.requires_grad_(False)
                # m.bias.requires_grad_(False)
            else :
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def set_bn_momentum(self, momentum): 
        self.layer[2].momentum = momentum

    def forward(self, X, X_raw) :
        encoding = self.layer(X)
        if self.ODL : 
            merged = torch.cat([X_raw, encoding], axis=1)
            probability = self.output(merged)
        else : 
            probability = self.output(encoding)

        return probability

    def transform(self, X):
        encoding = self.layer(X)
        return encoding

class EdSNNPreceptron(nn.Module):
    def __init__(self, features, raw_features, classes , nodes, device, gamma, beta, bn_learnable, track_running_stats , p_drop , ODL):
        super(EdSNNPreceptron, self).__init__()

        # init params
        self.features = features
        self.raw_features = raw_features
        self.classes = classes
        self.nodes = nodes
        self.d = device
        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats,
        self.p_drop = p_drop
        self.ODL = ODL # output direct link 
        # init layers
        self.layer = nn.Sequential(
            nn.Linear(self.features , self.nodes),
            nn.SELU(),
            nn.AlphaDropout(p=p_drop),)

        self.layer.apply(self.__init_weights__)
        
        # initialzie selu activation 
        for param in self.layer.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

        self.output = nn.Sequential(
                nn.Linear(self.nodes + self.raw_features if ODL else self.nodes , classes),
                nn.Softmax(dim=1))

    def __init_weights__(self, m):
        # initalize batchnorm data 
        if isinstance(m, nn.BatchNorm1d) and (m.weight is not None) and (m.bias is not None) :
            if  self.bn_learnable : 
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)
            else :
                m.weight.data.fill_(self.gamma)
                m.bias.data.fill_(self.beta)
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def set_bn_momentum(self, momentum): 
        self.layer[2].momentum = momentum

    def forward(self, X, X_raw) :
        encoding = self.layer(X)
        if self.ODL : 
            merged = torch.cat([X_raw, encoding], axis=1)
            probability = self.output(merged)
        else : 
            probability = self.output(encoding)
        return probability

    def transform(self, X):
        encoding = self.layer(X)
        return encoding

class EDBP():

    # BP version based on forward method
    def __init__(self, classes : int, nodes : int , layers : int = 5 , epochs : int = 10, learning_rate : float = 1e-3,  p_drop : float = 0  ,device  : str ='cpu',weight_decay : int =0, batch_percentage : int = 0.05, gamma : int  = 1 , beta : int = 0, bn_learnable : bool = True, track_running_stats : bool = True ,l1_weight : int = 0 , boost_lr : int = 0.1,  seed : int = -1, verbose : int =1  , boost = False , bootstrap = False , est_weight = False,  HDL = False , OHL = False , output_loc= None , snn = False , ep = 1e-3) :

        super().__init__()
        self.nodes = nodes 
        self.classes = classes
        self.d = device
        self.layers = layers
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_percentage = batch_percentage
        self.gamma = gamma
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats
        self.l1_weight = l1_weight
        self.seed = seed
        self.est_weight = est_weight # estimator weight
        self.boost = boost
        self.boost_lr = boost_lr 
        self.bootstrap = bootstrap
        self.p_drop = p_drop
        self.HDL = HDL # input direct link
        self.ODL = OHL # output direct link 
        self.verbose = verbose
        self.output_loc = output_loc
        self.snn = snn 
        self.ep = ep 
        self.state_init()
    
    def state_init(self):
        self.Params=EasyDict()
        self.Params['models'] = []
        self.Params['estimator_weights'] = []

        self.history = {} # store results 

    def __gen_bootstrap_index__(self , size , weights ):
        if self.bootstrap :
           return  torch.Tensor(list(WeightedRandomSampler(weights, size, replacement=True))).type(torch.long)
        else :
            return torch.arange(size)

    def train(self, X , y, X_val = None , y_val = None, epochs = None ):
        
        self.state_init()
        if self.seed != -1 :
            self.fix_seed(self.seed)

        if epochs is None :
            # repeat the self.epochs for each layer
            epochs = [self.epochs] * self.layers
        
        run_id =  datetime.datetime.now().strftime("%d_%m_%YT_%H_%M_%S")

        X = torch.Tensor(X).float().to(self.d)
        y = torch.Tensor(y).float().to(self.d)
        train_encoding = X.clone().to(self.d)
        train_sample_weight = torch.ones(X.shape[0]).to(self.d) / X.shape[0]
        
        # bootstraping index 
        bs_idx = self.__gen_bootstrap_index__(X.shape[0] , train_sample_weight)

        if not X_val is None : 
            X_val = torch.Tensor(X_val).float().to(self.d)
            y_val = torch.Tensor(y_val).float().to(self.d)
            val_encoding = X_val.clone().to(self.d)
            val_sample_weight = torch.ones(X_val.shape[0]).to(self.d) / X_val.shape[0]

        if self.batch_percentage != 0 : 
            n_steps = np.ceil(1 / self.batch_percentage)


        self.best_epochs = [] 

        for layer_id in range(int(self.layers)) :


            self.__consol__(f"Layer {layer_id}:\n{'='*20}" , level = 1)

            if self.snn : 
                model = EdSNNPreceptron(features = train_encoding.shape[1] ,raw_features = X.shape[1] , classes = y.shape[1] , nodes  = self.nodes ,device =  self.d ,
                        gamma=  self.gamma, beta= self.beta,  bn_learnable = self.bn_learnable,
                        track_running_stats=self.track_running_stats, ODL = self.ODL, p_drop=self.p_drop).to(self.d)

            else :
                model = EdPreceptron(features = train_encoding.shape[1] ,raw_features = X.shape[1] , classes = y.shape[1] , nodes  = self.nodes ,device =  self.d ,
                                    gamma=  self.gamma, beta= self.beta,  bn_learnable = self.bn_learnable,
                                    track_running_stats=self.track_running_stats, ODL = self.ODL, p_drop=self.p_drop).to(self.d)


            loss_fn = nn.CrossEntropyLoss(reduction='none')
            opt = torch.optim.Adam(model.parameters() , lr = self.lr, weight_decay=self.weight_decay)
            
            early_stopper = EarlyStopping(patience=10, smoothing=10 , checkpoint_name=f"{self.output_loc}/MLP_{run_id}.pt", delta=0)

            for epoch in range(epochs[layer_id]) :
                # set momentum in the first epoch
                if epoch == 0 : 
                    # change momentum ?
                    model.set_bn_momentum(momentum=0.2)

                self.train_loop(train_encoding[bs_idx],  X[bs_idx], y[bs_idx],
                                model , loss_fn , opt , train_sample_weight[bs_idx])

                train_loss , train_acc = self.__eval__(model , train_encoding[bs_idx], X[bs_idx] , y[bs_idx] , train_sample_weight[bs_idx] ,loss_fn)
                if not X_val is None :
                    val_loss , val_acc = self.__eval__(model , val_encoding, X_val , y_val , val_sample_weight, loss_fn)
                    self.register_history(layer_id, train_loss=train_loss , train_acc= train_acc , val_loss=val_loss, val_acc=val_acc)
                
                    self.__consol__(f"Epoch {epoch}, stps {n_steps} ->  train_loss = {train_loss:0.2f}, trian_acc = {train_acc * 100:0.2f}, val_loss = {val_loss :0.2f}, val_acc = {val_acc *100 :0.2f}", level=1)
                    # update sample weight
                    early_stopper(val_loss, model)
                    if early_stopper.early_stop:
                        self.__consol__(f"Early stopping at epoch {epoch}", level=1)
                        break           
                else :
                    self.register_history(layer_id, train_loss=train_loss , train_acc= train_acc , val_loss=None, val_acc=None)
                    self.__consol__(f"Epoch {epoch}, stps {n_steps} ->  train_loss = {train_loss:0.2f}, trian_acc = {train_acc * 100:0.2f}", level=1)

            
            if not X_val is None :            
                early_stopper.load_checkpoint(model)
                if early_stopper.early_stop : 
                    self.best_epochs.append(early_stopper.best_epoch)
                else : 
                    self.best_epochs.append(epochs[layer_id])

            self.Params.models.append(model)

            if self.boost or self.est_weight : 
                train_sample_weight , train_estimator_weight = self.__update_weights__(X, y, train_sample_weight , self.boost)
                if not X_val is None :
                    val_sample_weight , val_estimator_weight = self.__update_weights__(X_val, y_val, val_sample_weight, self.boost)
                    self.Params.estimator_weights.append(val_estimator_weight)
                else : 
                    self.Params.estimator_weights.append(train_estimator_weight)
            else : 
                # if no boosting and weighting is used, then use the same weight for all estimators
                self.Params.estimator_weights.append(torch.tensor(1.0).to(self.d))

            train_encoding = model.transform(train_encoding).detach().clone()
            if not X_val is None :
                val_encoding = model.transform(val_encoding)
        
            if self.HDL : 
                # append raw data to the encoding
                train_encoding = torch.cat((train_encoding , X) , dim=1)
                if not X_val is None :
                    val_encoding = torch.cat((val_encoding , X_val) , dim=1)
            if self.bootstrap : 
                bs_idx = self.__gen_bootstrap_index__(X.shape[0] , train_sample_weight)


        if not X_val is None : 
            return self.best_epochs , self.Params.estimator_weights
        else : 
            return None 

    def fix_seed(self , seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # essentialy, it need to be formated to calculate weight on the validation data not the training data to avoid overfitting. 
    # def __update_sample_weight__(self , model , encoding , X , y , sample_weight ):
    #     used for boosting
    #     y_pred = model(encoding , X)
    #     y_pred = y_pred.detach().cpu().clone()
    #     y_pred = y_pred.argmax(1)
    #     y_true = y.detach().cpu().clone()
    #     incorrect = y_pred != y_true.argmax(1)
    #     incorrect = incorrect.to(self.d)
    #     error_rate = torch.sum(sample_weight[incorrect]) / torch.sum(sample_weight)
    #     estimator_weight = torch.log((1 - error_rate) / error_rate).to(self.d)  + torch.log(torch.tensor(self.classes -1)).to(self.d) 
    #     sample_weight *= torch.exp(estimator_weight * incorrect)
    #     sample_weight /= torch.sum(sample_weight)

    #     return sample_weight, estimator_weight
    
    def __update_weights__(self  , X , y , sample_weight, boosting):
        print(sum(sample_weight))
        # used for only estimation weights     
        y_pred = self.predict(X)
        y_pred = torch.Tensor(y_pred)
        y_pred = y_pred.argmax(1)
        y_true = y.detach().cpu().clone()
        incorrect = y_pred != y_true.argmax(1)
        incorrect = incorrect.to(self.d)
        error_rate = torch.sum(sample_weight[incorrect]) / torch.sum(sample_weight)
        estimator_weight = torch.log((1 - (error_rate + self.ep)) / (error_rate +  self.ep)).to(self.d)  + torch.log(torch.tensor(self.classes -1)).to(self.d) 
        if boosting : 
            # update sample weight incase of boosting
            sample_weight *= torch.exp(estimator_weight * incorrect * self.boost_lr)
            sample_weight /= torch.sum(sample_weight)

        print(sum(sample_weight))
        return sample_weight, estimator_weight

    def register_history(self, layer_id , train_loss , train_acc , val_loss ,val_acc ) : 

        if not layer_id in self.history.keys():
            # initialzie history object 
            self.history[layer_id]= {
                    "train_loss" : [],
                    "train_acc" : [],
                    "val_loss" : [],
                    "val_acc" : [],
                }
        # store results 
        self.history[layer_id]['train_loss'].append(train_loss)
        self.history[layer_id]['train_acc'].append(train_acc)
        self.history[layer_id]['val_loss'].append(val_loss)
        self.history[layer_id]['val_acc'].append(val_acc)

    def train_loop(self , encoding , X , y, model , loss_fn , opt, weight) :
        model.train()

        batch_size =  int(np.ceil(self.batch_percentage * X.shape[0]))
        indexs = torch.randperm(X.shape[0]) # premutated to have random shuffling

        for batch_id in range(0 , X.shape[0] , batch_size) :
            batch_idx = indexs[batch_id : batch_id + batch_size ]
            train_pred = model(encoding[batch_idx], X[batch_idx])
            loss = loss_fn(train_pred , y[batch_idx])
            loss = (loss * weight[batch_idx] / weight[batch_idx].sum()).sum()
            # loss = loss.mean()
            # compute L1 loss
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.reshape(-1))
            l1 = self.l1_weight * torch.abs(torch.cat(l1_parameters)).sum()
            loss += l1

            opt.zero_grad()
            loss.backward()
            opt.step()

    def __eval__(self, model , encoding, X , y , weights, loss_fn):

        model.eval()
        with torch.no_grad():
            pred = model(encoding, X)
            loss = loss_fn(pred , y)
            loss = (loss * weights / weights.sum()).sum()
            # loss = loss.mean()
            loss = loss.detach().cpu().numpy()
            pred = pred.cpu().detach().numpy()
            acc = accuracy_score(y.argmax(1).cpu() , pred.argmax(1))

        return loss , acc 

    def set_estimator_weights(self, estimator_weights):
        self.Params.estimator_weights = estimator_weights

    def predict(self, X):
        X = torch.Tensor(X).to(self.d)
        scores = []
        encoding = X.clone().to(self.d)
        for i , model in enumerate(self.Params.models) :
            model.eval()
            pred_score = model(encoding, X)
            pred_score = pred_score.cpu().detach().numpy()
            if len(self.Params.estimator_weights) == i+1 : 
                estimator_weight = self.Params.estimator_weights[i].cpu().detach().numpy()
            else : 
                estimator_weight = 1
            scores.append(pred_score * estimator_weight)
            encoding = model.transform(encoding)
            if self.HDL :
                encoding = torch.cat((encoding , X) , dim=1)
        scores = np.array(scores)
        return scores.mean(0)

    def __consol__(self, string : str , level : int ) : 
        if level <= self.verbose : 
            print(string)
