# -*- coding: utf-8 -*-
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
from DLoader.uci import UCIDataset
import time 
import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade, Scenario
from pathlib import Path
import pickle as pkl
from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score
import random


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

class LWTEDBP():

    # BP version based on forward method
    def __init__(self, classes : int, epochs : int , gama : int = 1 , beta : int = 0 , bn_learnable : bool = True ,  track_running_stats : bool = True , device  : str ='cpu',weight_decay : int =0,  seed : int = -1, verbose : int =1  , boost = False , bootstrap = False , est_weight = False, conf_weights = False, output_loc= None , snn = False , ep = 1e-3 , min_batch_size = 10) :

        super().__init__()
        self.classes = classes
        self.d = device
        self.epochs = epochs 
        self.gama = gama 
        self.beta = beta
        self.bn_learnable = bn_learnable
        self.track_running_stats = track_running_stats
        self.min_batch_size = min_batch_size
        self.weight_decay = weight_decay
        self.est_weight = est_weight # estimator weight
        self.boost = boost
        self.bootstrap = bootstrap
        # self.ODL = OHL # output direct link 
        self.verbose = verbose
        self.output_loc = output_loc
        self.snn = snn 
        self.ep = ep 
        self.conf_weights = conf_weights 
        self.state_init()


    def state_init(self):
        self.Params=EasyDict()
        self.Params['models'] = []
        self.Params['HDL'] = []

        self.Params['estimator_weights'] = []

        self.history = {} # store results 

    def __gen_bootstrap_index__(self , size , weights ):
        if self.bootstrap :
           return  torch.Tensor(list(WeightedRandomSampler(weights, size, replacement=True))).type(torch.long)
        else :
            return torch.arange(size)

    def train_layer(self, X , y, X_val, y_val, params,train_sample_weight=None,val_sample_weight=None, append = False, seed = -1):
        
        if seed != -1 :
            self.fix_seed(seed)
        
        # calculate layer id 
        layer_id = len(self.Params.models)

        run_id =  datetime.datetime.now().strftime("%d_%m_%YT_%H_%M_%S")

        X = torch.Tensor(X).float().to(self.d)
        y = torch.Tensor(y).float().to(self.d)

        X_val = torch.Tensor(X_val).float().to( self.d)
        y_val = torch.Tensor(y_val).float().to( self.d)
        
        if train_sample_weight is None :
            train_sample_weight = torch.ones(X.shape[0]).to(self.d) / X.shape[0]
        if val_sample_weight is None :
            val_sample_weight = torch.ones(X_val.shape[0]).to(self.d) / X_val.shape[0]
        
        if len(self.Params.models) == 0 : 
            train_encoding = X.clone().to(self.d)
            val_encoding = X_val.clone().to(self.d)
        else : 
            train_encoding = self.transform(X)
            val_encoding = self.transform(X_val)
        # train_encoding = X.clone().to(self.d)
        # val_encoding = X_val.clone().to(self.d)

        # bootstraping index 
        bs_idx = self.__gen_bootstrap_index__(X.shape[0] , train_sample_weight)


        if params['batch_percentage'] != 0 : 
            n_steps = np.ceil(1 / params['batch_percentage'])


        self.best_epochs = [] 


        self.__consol__(f"Training Layer {layer_id}:\n{'='*20}" , level = 1)

        if self.snn : 
            model = EdSNNPreceptron(features = train_encoding.shape[1] ,raw_features = X.shape[1] , classes = y.shape[1] , nodes  = params['nodes'] ,device =  self.d ,
                    gamma= self.gama, beta= self.beta,  bn_learnable = self.bn_learnable,
                    track_running_stats=self.track_running_stats, ODL = params['ODL'], p_drop=params['p_drop']).to(self.d)

        else :
            model = EdPreceptron(features = train_encoding.shape[1] ,raw_features = X.shape[1] , classes = y.shape[1] , nodes  = params['nodes'] ,device =  self.d ,
                    gamma= self.gama, beta= self.beta,  bn_learnable = self.bn_learnable,
                    track_running_stats=self.track_running_stats, ODL = params['ODL'], p_drop=params['p_drop']).to(self.d)


        loss_fn = nn.CrossEntropyLoss(reduction='none')
        opt = torch.optim.Adam(model.parameters() , lr = params['learning_rate'], weight_decay=params['weight_decay'])
        
        early_stopper = EarlyStopping(patience=10, smoothing=10 , checkpoint_name=f"{self.output_loc}/MLP_{run_id}.pt", delta=0)

        for epoch in range(self.epochs) :
            # set momentum in the first epoch
            if epoch == 0 : 
                # change momentum ?
                model.set_bn_momentum(momentum=0.2)

            self.train_loop(train_encoding[bs_idx],  X[bs_idx], y[bs_idx],
                            model , loss_fn , opt , train_sample_weight[bs_idx], params['l1_weight'] , params['batch_percentage'])

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

        
        early_stopper.load_checkpoint(model)
        best_epoch = early_stopper.best_epoch if early_stopper.early_stop else epoch


        if append :
            # only store the current model if the append flag is true
            self.Params.models.append(model)
            self.Params.HDL.append(params['HDL'])
    
            if self.boost or self.est_weight : 
                train_sample_weight , train_estimator_weight = self.__update_weights__(X, y, train_sample_weight , self.boost, params['boost_lr'])
                val_sample_weight , val_estimator_weight = self.__update_weights__(X_val, y_val, val_sample_weight, self.boost, params['boost_lr'])
                self.Params.estimator_weights.append(val_estimator_weight)
            else : 
                # if no boosting and weighting is used, then use the same weight for all estimators
                self.Params.estimator_weights.append(torch.tensor(1.0).to(self.d))
        else : 
            
            if (self.boost or self.est_weight) and len(self.Params.models) > 0 : 

                train_sample_weight , train_estimator_weight = self.__update_weights__(X, y, train_sample_weight , self.boost, params['boost_lr'])
                val_sample_weight , val_estimator_weight = self.__update_weights__(X_val, y_val, val_sample_weight, self.boost, params['boost_lr'])
            else :
                train_estimator_weight = torch.tensor(1.0).to(self.d)
                val_estimator_weight = torch.tensor(1.0).to(self.d) 

        train_loss , train_acc = self.__eval__(model , train_encoding[bs_idx], X[bs_idx] , y[bs_idx] , train_sample_weight[bs_idx] ,loss_fn)
        val_loss , val_acc = self.__eval__(model , val_encoding, X_val , y_val , val_sample_weight, loss_fn)

        # if len(self.Params.models) == 0 :        
        #     train_loss , train_acc = self.__eval__(model , train_encoding[bs_idx], X[bs_idx] , y[bs_idx] , train_sample_weight[bs_idx] ,loss_fn)
        #     val_loss , val_acc = self.__eval__(model , val_encoding, X_val , y_val , val_sample_weight, loss_fn)

        # else : 
        #     if append : 
        #         train_pred = self.predict(X)
        #         train_acc = accuracy_score(y.argmax(1).cpu() , train_pred.argmax(1))

        #         val_pred =  self.predict(X_val)
        #         val_acc = accuracy_score(y_val.argmax(1).cpu() , val_pred.argmax(1))
        #     else : 
        #         # train_pred = self.__cumm_predict__(X , model , train_estimator_weight)
        #         train_pred = self.predict(X)
        #         train_acc = accuracy_score(y.argmax(1).cpu() , train_pred.argmax(1))

        #         # val_pred =  self.__cumm_predict__(X_val , model , val_sample_weight)
        #         val_pred =  self.predict(X_val)
        #         val_acc = accuracy_score(y_val.argmax(1).cpu() , val_pred.argmax(1))

        return model, best_epoch, train_acc , val_acc , train_sample_weight , val_sample_weight

    def fix_seed(self , seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def __update_weights__(self  , X , y , sample_weight, boosting, boosting_lr):
        # modify to work with the layer wise tunning. 
        print(sum(sample_weight))
        # used for only estimation weights     
        y_pred_score = self.predict(X)
        y_pred_score = torch.Tensor(y_pred_score)
        y_pred = y_pred_score.argmax(1)
        y_true = y.detach().cpu().clone()
        incorrect = y_pred != y_true.argmax(1)
        incorrect = incorrect.to(self.d)
        error_rate = torch.sum(sample_weight[incorrect]) / torch.sum(sample_weight)
        estimator_weight = torch.log((1 - (error_rate + self.ep)) / (error_rate +  self.ep)).to(self.d)  + torch.log(torch.tensor(self.classes -1)).to(self.d) 
        if boosting : 
            # update sample weight incase of boosting
            if self.conf_weights : 
                conf_weights = self.__conf_weights__(y_pred_score , y_true)
                conf_weights = conf_weights.to(self.d)
                # sample_weight += conf_weights
                sample_weight *= torch.exp(estimator_weight * conf_weights * boosting_lr)


            else : 
                sample_weight *= torch.exp(estimator_weight * incorrect * boosting_lr)
            
            sample_weight /= torch.sum(sample_weight)

        print(sum(sample_weight))
        return sample_weight, estimator_weight

    def __conf_weights__(self, prediction, true):
        correct_score = torch.max(prediction*true,1).values
        wrong_max_score = torch.max(prediction*(1-true),1).values
        # mask = correct_score > wrong_max_score
        weight = 1 / torch.exp(correct_score-wrong_max_score)
        return weight

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

    def train_loop(self , encoding , X , y, model , loss_fn , opt, weight, l1_weight , batch_percentage ) :
        model.train()

        batch_size =  max(int(np.ceil(batch_percentage * X.shape[0])), self.min_batch_size)
        indexs = torch.randperm(X.shape[0]) # premutated to have random shuffling

        for batch_id in range(0 , X.shape[0] , batch_size) :
            batch_idx = indexs[batch_id : batch_id + batch_size ]
            if batch_idx.size()[0] <= 1:
                continue
            train_pred = model(encoding[batch_idx], X[batch_idx])
            loss = loss_fn(train_pred , y[batch_idx])
            loss = (loss * weight[batch_idx] / weight[batch_idx].sum()).sum()
            # loss = loss.mean()
            # compute L1 loss
            l1_parameters = []
            for parameter in model.parameters():
                l1_parameters.append(parameter.reshape(-1))
            l1 = l1_weight * torch.abs(torch.cat(l1_parameters)).sum()
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
            if self.Params.HDL[i] :
                encoding = torch.cat((encoding , X) , dim=1)
        scores = np.array(scores)
        return scores.mean(0)
    
    def __cumm_predict__(self, X, current_model , cm_ew):
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
            if self.Params.HDL[i] :
                encoding = torch.cat((encoding , X) , dim=1)
        
        with torch.no_grad():
            pred_score = current_model(encoding, X)
            pred_score = pred_score.cpu().detach().numpy()
            cm_ew = cm_ew.cpu().detach().numpy()
            scores.append(pred_score * estimator_weight)
        
        scores = np.array(scores)
        return scores.mean(0)
    
    def transform(self, X):
        X = torch.Tensor(X).to(self.d)
        encoding = X.clone().to(self.d)
        for i , model in enumerate(self.Params.models) :
            model.eval()
            with torch.no_grad():
                encoding = model.transform(encoding)
            if self.Params.HDL[i] :
                encoding = torch.cat((encoding , X) , dim=1)
        return encoding
    
    def __consol__(self, string : str , level : int ) : 
        if level <= self.verbose : 
            print(string)

