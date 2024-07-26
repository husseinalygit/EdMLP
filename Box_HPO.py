# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:44:33 2022

@author: Hussein Aly
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time
import torch
import pickle as pkl
import os
from DLoader.uci import UCIDataset
from scipy import stats
from sklearn.model_selection import RepeatedKFold
import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade, Scenario
from pathlib import Path
from Models.backprob import MLP_model , EDBP
from Models.edRVFL import  BP_WPRVFL
from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score
from torch.multiprocessing import Pool, set_start_method
import datetime 
import evaluators
import argparse  # Import argparse module

# optimizers 

def EdMLP_optimizer_wraper(dloader, check_point_loc,  reps = 5 , device='cpu') :
    def EdMLP_optimizer(config , seed = 41):
        config_dict = dict(config)
        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight=False , boost = False , HDL=False , OHL=False, output_loc =check_point_loc, snn =False)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdMLP_optimizer

def EdSNN_boost_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdSNN_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = config_dict['boost_lr'] , HDL=False , OHL=False, output_loc =check_point_loc , snn = True)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdSNN_optimizer

def EdMLP_boost_optimizer_wraper(dloader, check_point_loc,  reps = 5 , device='cpu') :
    def EdMLP_optimizer(config , seed = 41):
        config_dict = dict(config)
        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = config_dict['boost_lr'] , HDL=False , OHL=False, output_loc =check_point_loc, snn =False)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdMLP_optimizer

def EdSNN_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdSNN_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = False , boost = False , HDL=False , OHL=False, output_loc =check_point_loc , snn = True)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdSNN_optimizer

def EdMLP_DLHO_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdMLP_DLHO_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = False , boost = False , HDL=True , OHL=True, output_loc =check_point_loc, snn =False)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdMLP_DLHO_optimizer

def EdSNN_DLHO_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdSNN_DLHO_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = False , boost = False , HDL=True , OHL=True, output_loc =check_point_loc , snn = True)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdSNN_DLHO_optimizer

def EdMLP_DLHO_boost_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdMLP_DLHO_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = config_dict['boost_lr'], HDL=True , OHL=True, output_loc =check_point_loc, snn =False)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdMLP_DLHO_optimizer

def EdRVFL_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdRVFL_optimizer(config , seed = 41):


        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                bb_rvfl = BP_WPRVFL(classes=y_train.shape[1],
                                args={'C':config_dict['C'], 'N':config_dict['N'] } ,
                                layers=config_dict['layers'] , epochs = 0,
                                learning_rate = 0 , low=1,
                                prune_rate = 0, weight_decay = 0,
                                batch_percentage = 1, gamma=config_dict['gamma'],
                                beta=config_dict['beta'],bn_affine=True,
                                device=device, seed = seed)
                

                train_pred , val_pred, layer_train_acc, layer_val_acc = bb_rvfl.train(X=torch.tensor(X_train).float(),
                y=torch.tensor(y_train).float(),
                X_val = torch.tensor(X_val).float(),
                y_val = torch.tensor(y_val).float())

                train_pred = np.array(train_pred)
                val_pred = np.array(val_pred)

                train_mv_acc = accuracy_score(y_train.argmax(1) , train_pred.sum(0).argmax(1))
                test_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.sum(0).argmax(1))


                avg_acc.append(test_mv_acc)
                del bb_rvfl
                torch.cuda.empty_cache()

        return 1-np.mean(avg_acc)
    return EdRVFL_optimizer

def EdSNN_DLHO_boost_optimizer_wraper(dloader, check_point_loc, reps = 5, device='cpu') :

    def EdSNN_DLHO_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                edbp = EDBP(classes=y_train.shape[1], nodes = config_dict['nodes']  ,
                            layers=config_dict['layers'] , epochs = 100,
                            learning_rate = config_dict['learning_rate'],
                            weight_decay = config_dict['weight_decay'],
                            l1_weight = config_dict['l1_weight'], 
                            batch_percentage = config_dict['batch_percentage'], device=device,
                            gamma = 1 , beta=0, p_drop = config_dict['p_drop'],
                            bn_learnable=True, track_running_stats = True, seed = seed ,
                            verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = config_dict['boost_lr'], HDL=True , OHL=True, output_loc =check_point_loc , snn = True)


                edbp.train(X=X_train,
                            y=y_train,
                            X_val = X_val,
                            y_val = y_val)

                val_pred = edbp.predict(X_val) 

                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))

                avg_acc.append(val_mv_acc)
                del edbp
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return EdSNN_DLHO_optimizer

def MLP_optimizer_wraper(dloader,  check_point_loc, reps = 5, device='cpu' ,) :

    def MLP_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                mlModel = MLModel(classes=y_train.shape[1], nodes = config_dict['nodes'],
                                layers = config_dict['layers'], epochs = 100,
                                learning_rate = config_dict['learning_rate'],  l1_weight = config_dict['l1_weight'],
                                device = device, batch_percentage = config_dict['batch_percentage'],
                                weight_decay = config_dict['weight_decay'] , direct_link = False ,
                                seed = seed, verbose = 1 , output_loc = check_point_loc , p_drop = config_dict['p_drop'] , snn = False)


                mlModel.train(X=torch.tensor(X_train).float(),
                                        y=torch.tensor(y_train).float(), 
                                        X_val= torch.tensor(X_val).float(),
                                            y_val=torch.tensor(y_val).float(), ) 
                train_pred = mlModel.predict(X_train)
                val_pred = mlModel.predict(X_val)

                train_pred = np.array(train_pred)
                val_pred = np.array(val_pred)

                train_mv_acc = accuracy_score(y_train.argmax(1) , train_pred.argmax(1))
                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))
                print(val_mv_acc)
                avg_acc.append(val_mv_acc)
                del mlModel
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return MLP_optimizer

def SNN_optimizer_wraper(dloader,  check_point_loc, reps = 5 , device='cpu' ,) :

    def SNN_optimizer(config , seed = 41):

        config_dict = dict(config)

        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                mlModel = MLModel(classes=y_train.shape[1], nodes = config_dict['nodes'],
                                layers = config_dict['layers'], epochs = 100,
                                learning_rate = config_dict['learning_rate'],  l1_weight = config_dict['l1_weight'],
                                device = device, batch_percentage = config_dict['batch_percentage'],
                                weight_decay = config_dict['weight_decay'] , direct_link = False ,
                                seed = seed, verbose = 1 , output_loc = check_point_loc , p_drop = config_dict['p_drop'] , snn = True)


                mlModel.train(X=torch.tensor(X_train).float(),
                                        y=torch.tensor(y_train).float(), 
                                        X_val= torch.tensor(X_val).float(),
                                            y_val=torch.tensor(y_val).float(), ) 
                train_pred = mlModel.predict(X_train)
                val_pred = mlModel.predict(X_val)

                train_pred = np.array(train_pred)
                val_pred = np.array(val_pred)

                train_mv_acc = accuracy_score(y_train.argmax(1) , train_pred.argmax(1))
                val_mv_acc = accuracy_score(y_val.argmax(1) , val_pred.argmax(1))
                print(val_mv_acc)
                avg_acc.append(val_mv_acc)
                del mlModel
                torch.cuda.empty_cache()
        return 1-np.mean(avg_acc)
    return SNN_optimizer


# utility functions 

def configspace_from_map(conf_map, seed=41) :

    conf_space = CS.ConfigurationSpace(seed=seed)
    conf_space.add_hyperparameters(conf_map)

    return conf_space

def optimize(hp_config ,output_loc, opt_function,n_trials = 100, walltime_limit= np.inf ):
    
    scenario = Scenario(hp_config,
                     output_directory=Path(output_loc),
                     walltime_limit=walltime_limit,  # Limit to max one hour
                     n_trials=n_trials,
                     n_workers = 1)

    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario)
    smac = HyperparameterOptimizationFacade(scenario,
                                         opt_function,
                                         initial_design=initial_design,
                                         overwrite=True)

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(hp_config.get_default_configuration())
    print(f"Default cost: {default_cost}")
        
    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"incumbent cost: {incumbent_cost}")

    best_dict = { "target" : 1-incumbent_cost ,
                "params" : dict(incumbent)}

    return best_dict

def run_optimization_evaluation(dataset_name , model_name, bounds_map , wraper_mape, device , run_id , n_trials = 100 , tunning_reps=5 ,  eval_reps= 5, random_seed = 41): 
    try :
        print(f"optimize -> {dataset_name} on  {model_name}")
        loader = UCIDataset(dataset_name, parent="DLoader/UCIdata")

        start_time = time.time()

        output_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/logs/"

        check_point_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/checkpoints/"
        
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)

        if not os.path.exists(check_point_loc):
            os.makedirs(check_point_loc)


        best_dict = optimize(configspace_from_map(bounds_map[model_name]),
                            output_loc, wraper_mape[model_name](loader , check_point_loc , reps = tunning_reps , device = device), n_trials = n_trials)

        with open(f"{output_loc}/../best_{run_id}.pk" , 'wb') as file :
            pkl.dump(best_dict , file)
        
        metrics = evaluators.model_evaluator(loader , best_dict['params'] , evaluators.model_func_map[model_name], check_point_loc ,  eval_reps , device , console = False , random_seed = random_seed)



        ds_time = time.time() - start_time 
        print(f"finisehd -> {dataset_name} on  {model_name}, took {ds_time // 60} miuntes")

        metrics.append(ds_time)

        return  metrics 
    except Exception as e : 
        print(e)
        return e 

if __name__ == "__main__" :
    supported_models = ["EdMLP" , "EdSNN" , "EdMLP_DLHO" , "EdSNN_DLHO" , "MLP" , "SNN" , "EdMLP_Boost" , "EdSNN_Boost" , "EdMLP_DLHO_Boost" , "EdSNN_DLHO_Boost" , "EdRVFL"]
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization Script')
    # Add arguments
    parser.add_argument('--dataset', type=str, default=None, nargs = "+" , help='Name of the UCI dataset/s to use')
    parser.add_argument('--model', type=str, required=True, nargs="+", choices=supported_models, help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--tunning_reps', type=int, default=5, help='Number of repetitions for bayesian optimization')
    parser.add_argument('--eval_reps', type=int, default=5, help='Number of repetitions for evaluation')
    parser.add_argument('--random_state', type=int, default=41, help='Random state')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    # add run id 
    parser.add_argument('--run_id', type=str, default=None, help='Run ID, set the value only if you want to run multiple optimziation rounds with the same id. Otherwise leave blank')

    # Parse arguments
    args = parser.parse_args()

    shared_bounds = [
                    CS.UniformIntegerHyperparameter("layers",lower=1,upper=15, default_value=5),
                    CS.UniformIntegerHyperparameter("nodes",lower=128,upper=1024, default_value=256),
                    CS.UniformFloatHyperparameter("weight_decay",lower=1e-7,upper=1e-2,default_value=1e-5 , log=True),
                    CS.UniformFloatHyperparameter("learning_rate" , lower=1e-5 , upper=1e-1,default_value=1e-3 ,log=True) ,
                    CS.UniformFloatHyperparameter("l1_weight",lower=1e-7,upper=1e-2,default_value=1e-5 , log=True),
                    CS.UniformFloatHyperparameter("batch_percentage",lower=0.05,upper=0.5, default_value=0.05),
                    CS.UniformFloatHyperparameter("p_drop",lower=0,upper=0.5, default_value=0.05),]
        
    boost_params = [CS.UniformFloatHyperparameter("boost_lr",lower=1e-5,upper=1, default_value=1e-3 , log=True)]

    EdRVFL_bounds = [CS.UniformIntegerHyperparameter("C",lower=-12,upper=12, default_value=0),
                     CS.UniformIntegerHyperparameter("layers",lower=1,upper=10, default_value=5),
                     CS.UniformIntegerHyperparameter("N",lower=20,upper=1000, default_value=200),
                     CS.UniformFloatHyperparameter("gamma" , lower=0.5, upper=2 , default_value=0.5),
                     CS.UniformFloatHyperparameter("beta" , lower=-2, upper=2 , default_value=0.5)]
    
    bounds_map = {
      "EdMLP" : shared_bounds ,
      "EdSNN" : shared_bounds ,
      "EdMLP_DLHO" : shared_bounds ,
      "EdSNN_DLHO" : shared_bounds ,
      "MLP" :  shared_bounds ,  
      "SNN" : shared_bounds,
      "EdMLP_Boost" : shared_bounds + boost_params,
      "EdSNN_Boost" : shared_bounds + boost_params,
      "EdMLP_DLHO_Boost" : shared_bounds + boost_params,
      "EdSNN_DLHO_Boost" : shared_bounds + boost_params,
      "EdRVFL" : EdRVFL_bounds, 
    }

    wraper_mape = {
        "EdMLP" : EdMLP_optimizer_wraper,
        "EdSNN" : EdSNN_optimizer_wraper,
        "EdMLP_DLHO" : EdMLP_DLHO_optimizer_wraper ,
        "EdSNN_DLHO" :EdSNN_DLHO_optimizer_wraper,
        "MLP" : MLP_optimizer_wraper, 
        "SNN" : SNN_optimizer_wraper,
        "EdMLP_Boost" : EdMLP_boost_optimizer_wraper,
        "EdSNN_Boost" : EdSNN_boost_optimizer_wraper,
        "EdMLP_DLHO_Boost" : EdMLP_DLHO_boost_optimizer_wraper,
        "EdSNN_DLHO_Boost" : EdSNN_DLHO_boost_optimizer_wraper,
        "EdRVFL" : EdRVFL_optimizer_wraper
     }

    if args.dataset is None :
        datasets = ["abalone" , "arrhythmia" , "cardiotocography-10clases" , "cardiotocography-3clases" , "chess-krvkp"  , "congressional-voting" , "contrac" , "glass" , "molec-biol-splice" , "monks-3" , "musk-2" ,"oocytes_trisopterus_states_5b" , "spambase" , "statlog-image" , "statlog-landsat" ,"wall-following" , "waveform" , "waveform-noise", "breast-cancer-wisc-prog" , "breast-tissue" , "conn-bench-sonar-mines-rocks" , "conn-bench-vowel-deterding" , "hill-valley" , "ionosphere" , "iris" , "oocytes_merluccius_nucleus_4d" , "oocytes_merluccius_states_2f" , "oocytes_trisopterus_nucleus_2f" , "oocytes_trisopterus_states_5b" , "parkinsons" , "plant-shape" , "ringnorm" ,  "seeds" , "synthetic-control" , "twonorm" , "vertebral-column-2clases" , "vertebral-column-3clases"]
    else : 
        datasets = args.dataset

    if all([model in supported_models for model in args.model]) :
        models = args.model
    else :
        Exception(f"Model {args.model} not supported, supported models are {supported_models}")
    
    device = args.device
    
    n_trials = args.n_trials
    random_state= args.random_state

    if args.run_id is None : 
        run_id =  datetime.datetime.now().strftime("%d_%m_%YT_%H_%M")
    else : 
        run_id = args.run_id

    session_id = datetime.datetime.now().strftime("%d_%m_%YT_%H_%M")

    print(f"Run ID : {run_id}")
    print(f"Session ID : {session_id}")

    if not os.path.exists("results") : 
        os.makedirs("results")
    
    pool = Pool(args.n_jobs)
    pool_requests = {}
    for dataset_name in datasets : 
        pool_requests[dataset_name] = []
        for model_name in models :
            try : 
                pool_res = pool.apply_async(run_optimization_evaluation, (dataset_name , model_name , bounds_map , wraper_mape, device, run_id, n_trials , args.tunning_reps, args.eval_reps, random_state ) )
                pool_requests[dataset_name].append(pool_res)
            except Exception as e : 
                print(e)
    
    total_res = pd.DataFrame(columns=["train_acc",  "train_f1", "train_precision",  "train_recall",  "val_acc",  "val_f1", "val_precision",  "val_recall" , "test_acc", "test_f1", "test_precision", "test_recall", "model" , "dataset" ,  "duration"  ])
    
    for dataset_name in pool_requests.keys(): 
        for request, model_name in zip(pool_requests[dataset_name], models) :
            try : 
                request.wait()
                result =  request.get()
                if not result is None :
                    train_acc,  train_f1, train_precision,  train_recall,  val_acc,  val_f1, val_precision,  val_recall , test_acc, test_f1, test_precision, test_recall , duration = result
                    
                    res_con = np.concatenate((np.expand_dims(train_acc , 1), 
                                              np.expand_dims(train_f1 ,1),
                                              np.expand_dims(train_precision ,1),
                                              np.expand_dims(train_recall ,1),
                                              np.expand_dims(val_acc ,1),
                                              np.expand_dims(val_f1 ,1),
                                              np.expand_dims(val_precision ,1),
                                              np.expand_dims(val_recall ,1),
                                              np.expand_dims(test_acc ,1),
                                              np.expand_dims(test_f1 ,1),
                                              np.expand_dims(test_precision ,1),
                                              np.expand_dims(test_recall ,1),
                                              ) , 1)
                    
                    results_df = pd.DataFrame(res_con, columns = ["train_acc",  "train_f1", "train_precision",  "train_recall",  "val_acc",  "val_f1", "val_precision",  "val_recall" , "test_acc", "test_f1", "test_precision", "test_recall" ])
                    
                    results_df['model'] = model_name 
                    results_df['dataset'] = dataset_name
                    results_df['duration'] = duration 
                    
                    total_res = pd.concat((total_res , results_df ) )
                    total_res.to_csv(f"results/results_r{run_id}_s{session_id}.csv", index=False)
            except Exception as e : 
                print(e)
                
    pool.close()
    pool.join()
