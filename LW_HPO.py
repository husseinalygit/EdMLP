# -*- coding: utf-8 -*-
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
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from Models.LWRVFL import LWTEDBP
from torch.multiprocessing import Pool, set_start_method
import argparse


def classification_eval(true, predict, tag = None , consol=False): 

    acc = accuracy_score(true , predict)
    f1 = f1_score(true , predict, average="macro" , zero_division=0)
    precision = precision_score(true , predict, average="macro" , zero_division=0)
    recall = recall_score(true , predict, average="macro" , zero_division=0)

    if consol : 
        if not tag is None  : 
            print(tag)
        print(f"acc : {acc * 100 : 0.2f}")
        print(f"f1 : {f1 * 100 : 0.2f}")
        print(f"precision : {precision * 100 : 0.2f}")
        print(f"recall : {recall * 100 : 0.2f}")

    return acc , f1 , precision , recall 

def gen_ran_arr(size , seed=time.time(),  max= 1000 ): 
    random.seed(seed)
    arr = np.array([ int(random.random() * max) for i in range(size)])
    return arr 

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

def optimizer_wraper(dloader , model_ref , reps = 5 , train_sample_weight = None , val_sample_weight = None  ):
    def optimizer(config , seed = 41):
        config_dict = config.get_dictionary()
        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :
                print(f"Working on model {model_ref[split_idx]}")
                _, best_epoch, train_acc , val_acc, _ , _ = model_ref[split_idx].train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = config_dict, append = False , train_sample_weight=train_sample_weight[split_idx],val_sample_weight=val_sample_weight[split_idx] , seed =seed )
                print(val_acc)
                avg_acc.append(val_acc)
        return 1-np.mean(avg_acc)
    return optimizer

def nodes_optimizer_wraper(dloader , base_params  , check_point_loc,  reps = 5, device='cuda' ):
    def optimizer(config , seed = 41):
        config_dict = config.get_dictionary()
        # extend base params with config dict
        params = [{**layer_param , **config_dict} for layer_param in base_params]
        n_layers = len(params)
        avg_acc = []
        cv_splits = dloader.n_CV
        for split_idx in range(cv_splits) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _  = dloader.getitem(split_idx)
            for rep in range(reps) :

                torch.cuda.empty_cache()

                model = make_model(classes=dloader.n_types , epochs=epochs, device=device, check_point_loc=check_point_loc)
                
                train_weights = None
                val_weights = None

                for layer_id in range(n_layers) :
                    _, _, _ , _ , layer_train_weights , layer_val_weights = model.train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = params[layer_id], train_sample_weight = train_weights , val_sample_weight = val_weights , append = True , seed = seed )

                    train_weights = layer_train_weights
                    val_weights = layer_val_weights       

                val_pred = model.predict(X_val)
                val_metrics = classification_eval(y_val.argmax(1) , val_pred.argmax(1) , consol=False)
                val_acc = val_metrics[0]

                print(val_acc)
                avg_acc.append(val_acc)
        return 1-np.mean(avg_acc)
    return optimizer

def make_model( classes ,epochs , device , check_point_loc ) : 
    model = LWTEDBP(classes=classes, epochs = epochs , gama = 1 , beta = 0 , bn_learnable=True , track_running_stats=True , verbose =2, est_weight = True ,  conf_weights = True ,boost = True , bootstrap = False, output_loc =check_point_loc , snn = False, device=device)
    return model

def evaluate(dloader , params , n_layers , epochs ,  check_point_loc ,   reps = 5 , device='cpu' , console = False , random_seed = 41 ): 
    metrics = []
    seeds = gen_ran_arr(reps , random_seed )

    cv_splits = dloader.n_CV
    for cv_idx in range(cv_splits) :
        X_train, y_train, X_val, y_val , X_test , y_test , _ , _ = dloader.getitem(cv_idx)
        for rep_idx in range(reps):
            torch.cuda.empty_cache()

            model = make_model(classes=dloader.n_types , epochs=epochs, device=device, check_point_loc=check_point_loc)
            
            train_weights = None
            val_weights = None

            for layer_id in range(n_layers) :
                _, _, _ , _ , layer_train_weights , layer_val_weights = model.train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = params[layer_id], train_sample_weight = train_weights , val_sample_weight = val_weights , append = True, seed = seeds[rep_idx])

                train_weights = layer_train_weights
                val_weights = layer_val_weights       

            train_pred = model.predict(X_train) 
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test) 

            train_metrics = classification_eval(y_train.argmax(1) , train_pred.argmax(1), tag = f"Rep {rep_idx} -> train:\n{'='*20}" , consol=console)
            val_metrics = classification_eval(y_val.argmax(1) , val_pred.argmax(1), tag = f"Rep {rep_idx} -> val:\n{'='*20}" , consol=console)
            test_metrics = classification_eval(y_test.argmax(1) , test_pred.argmax(1), tag =f"Rep {rep_idx} -> test:\n{'='*20}" ,  consol=console)
            metrics.append([*train_metrics , *val_metrics,  *test_metrics])
            del model
            torch.cuda.empty_cache()

    return np.array(metrics).T.tolist()

def layer_wise_optimization(dataset_name, model_name , bounds , device , run_id , n_trials = 100 , epochs = 200, max_layers = 20 , layer_patience=2 , tunning_reps=5 ,  eval_reps= 5, random_seed = 41):
    print(n_trials)
    print(epochs)
    print(f"optimize -> {dataset_name} on  {model_name}")
    loader = UCIDataset(dataset_name, parent="DLoader/UCIdata")

    start_time = time.time()

    output_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/logs/"

    check_point_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/checkpoints/"
    
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    if not os.path.exists(check_point_loc):
        os.makedirs(check_point_loc)

    # create multiple model, one for each cross validation split. 
    models = [make_model(classes=loader.n_types , epochs=epochs, device=device, check_point_loc=check_point_loc) for i in range(loader.n_CV)]

    best_score = 0 
    layer_buffer = 0 # for layer patience 
    layer_params= []
    train_weights = [None for i in range(loader.n_CV) ] # for layer patience (train weights
    val_weights = [None for i in range(loader.n_CV) ] # for layer patience (val weights
    for layer_id in range(max_layers): 
        # for each layer find the best parameters 
        best_dict = optimize(configspace_from_map(bounds),
                        output_loc, optimizer_wraper(loader , model_ref = models ,reps = tunning_reps  , train_sample_weight = train_weights , val_sample_weight = val_weights), n_trials = n_trials)
        layer_params.append(best_dict['params'])

        # add the layer with the best found parameters to the model 
        for split_idx in range(loader.n_CV) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _ = loader.getitem(split_idx)
            _, _, _ , _ , layer_train_weights , layer_val_weights = models[split_idx].train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = layer_params[layer_id], train_sample_weight = train_weights[split_idx] , val_sample_weight = val_weights[split_idx], append = True , seed = random_seed )
            train_weights[split_idx] = layer_train_weights
            val_weights[split_idx] = layer_val_weights
            

        # calculate the model performance
        metrics = evaluate(dloader = loader , params = layer_params , n_layers = layer_id+1, epochs = epochs, check_point_loc = check_point_loc , reps= eval_reps , device =device , console = True , random_seed = random_seed)
        
        # if accuracy is not improved for layer_patience number of layers, stop the layer wise optimization
        if np.mean(metrics[-4]) > best_score :
            best_score = np.mean(metrics[-4])
            layer_buffer = 0
        else : 
            layer_buffer += 1
        if layer_buffer >= layer_patience : 
            break
    # remove the layers that did not improve the accuracy
    if layer_buffer != 0 : 
        layer_params = layer_params[:-layer_buffer]

    # final evalaution of the model perforamnce 
    metrics = evaluate(dloader = loader , params = layer_params , n_layers = len(layer_params), epochs = epochs, check_point_loc = check_point_loc , reps= eval_reps , device =device , console = True , random_seed = random_seed)

    params = {
        "target" : np.mean(metrics[-4]) , 
        "params" : layer_params
    }

    # save the best layer params
    with open(f"{output_loc}/../best_{run_id}.pk" , 'wb') as file :
        pkl.dump(params , file)



    ds_time = time.time() - start_time 

    print(f"finisehd -> {dataset_name} on  {model_name}, took {ds_time // 60} miuntes")

    metrics.append(ds_time)

    return  metrics 

def box_nodes_layer_wise_optimization(dataset_name, model_name , bounds , device , run_id , n_trials = 100 , epochs = 200, max_layers = 20 , layer_patience=2 , tunning_reps=5 ,  eval_reps= 5, random_seed = 41):
    # perform layer wise optimization for the given dataset and model, with the given bounds and hyperparameters.
    # then do another optimization to find the number of nodes in the model. 
    print(n_trials)
    print(epochs)
    print(f"optimize -> {dataset_name} on  {model_name}")
    loader = UCIDataset(dataset_name, parent="DLoader/UCIdata")

    start_time = time.time()

    output_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/logs/"

    check_point_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/checkpoints/"
    
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    if not os.path.exists(check_point_loc):
        os.makedirs(check_point_loc)

    # create multiple model, one for each cross validation split. 
    models = [make_model(classes=loader.n_types , epochs=epochs, device=device, check_point_loc=check_point_loc) for i in range(loader.n_CV)]

    best_score = 0 
    layer_buffer = 0 # for layer patience 
    layer_params= []
    train_weights = [None for i in range(loader.n_CV) ] # for layer patience (train weights
    val_weights = [None for i in range(loader.n_CV) ] # for layer patience (val weights
    for layer_id in range(max_layers): 
        # for each layer find the best parameters 
        best_dict = optimize(configspace_from_map(bounds),
                        output_loc, optimizer_wraper(loader , model_ref = models , reps = tunning_reps  , train_sample_weight = train_weights , val_sample_weight = val_weights), n_trials = n_trials)
        layer_params.append(best_dict['params'])

        # add the layer with the best found parameters to the model 
        for split_idx in range(loader.n_CV) :
            X_train, y_train, X_val, y_val , _ , _ , _ , _ = loader.getitem(split_idx)
            _, _, _ , _ , layer_train_weights , layer_val_weights = models[split_idx].train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = layer_params[layer_id], train_sample_weight = train_weights[split_idx] , val_sample_weight = val_weights[split_idx], append = True , seed = random_seed )
            train_weights[split_idx] = layer_train_weights
            val_weights[split_idx] = layer_val_weights
            

        # calculate the model performance
        metrics = evaluate(dloader = loader , params = layer_params , n_layers = layer_id+1, epochs = epochs, check_point_loc = check_point_loc , reps= eval_reps , device =device , console = True , random_seed = random_seed)
        
        # if accuracy is not improved for layer_patience number of layers, stop the layer wise optimization
        if np.mean(metrics[-4]) > best_score :
            best_score = np.mean(metrics[-4])
            layer_buffer = 0
        else : 
            layer_buffer += 1
        if layer_buffer >= layer_patience : 
            break
        
    
    # remove the layers that did not improve the accuracy
    if layer_buffer != 0 : 
        layer_params = layer_params[:-layer_buffer]
    
    
    # do another round of optimization to fine the number of nodes in the model, all layers will have the same number of nodes. 
    nodes_bounds = [bound for bound in bounds if bound.name == "nodes"]
    
    other_param_name = np.setdiff1d(list(layer_params[0].keys()) , ['nodes']) 
    other_param_val = [{param : lp[param] for param in other_param_name} for lp in layer_params]
    
    nodes_best_dict = optimize(configspace_from_map(nodes_bounds),
                output_loc, nodes_optimizer_wraper(loader , other_param_val , check_point_loc, reps = tunning_reps, device=device ), n_trials = 2)


    node_params = [{**lp, **nodes_best_dict['params']} for lp in layer_params]

    # final evalaution of the model perforamnce 
    metrics = evaluate(dloader = loader , params = node_params , n_layers = len(node_params), epochs = epochs, check_point_loc = check_point_loc , reps= eval_reps , device =device , console = True , random_seed = random_seed)


    params = {
        "target" : np.mean(metrics[-4]) , 
        "params" : node_params
    }

    # save the best layer params
    with open(f"{output_loc}/../best_{run_id}.pk" , 'wb') as file :
        pkl.dump(params , file)



    ds_time = time.time() - start_time 

    print(f"finisehd -> {dataset_name} on  {model_name}, took {ds_time // 60} miuntes")

    metrics.append(ds_time)

    return  metrics 


if __name__ == "__main__" : 

    supported_models = ["LEdMLP_DLHO_Boost", "BLEdMLP_DLHO_Boost"]

    # Define the argument parser
    parser = argparse.ArgumentParser(description='LayerWise Hyperparameter Optimization Script')
    # Add arguments
    parser.add_argument('--dataset', type=str, default=None, nargs = "+" , help='Name of the UCI dataset/s to use')
    parser.add_argument('--model', type=str, required=True, nargs="+", choices=supported_models, help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--tunning_reps', type=int, default=5, help='Number of repetitions for Bayesian optimization')
    parser.add_argument('--eval_reps', type=int, default=5, help='Number of repetitions for evaluation')
    parser.add_argument('--random_state', type=int, default=41, help='Random state')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    parser.add_argument('--layer_patience', type=int, default=2, help='Number of layers to wait before stopping layer wise optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--max_layers', type=int, default=15, help='Maximum number of layers to train')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID, set the value only if you want to run multiple optimziation rounds with the same id. Otherwise leave blank')

    # Parse arguments
    args = parser.parse_args()

    

    bounds = [
                    CS.UniformIntegerHyperparameter("nodes",lower=128,upper=1024, default_value=256),
                    CS.UniformFloatHyperparameter("weight_decay",lower=1e-7,upper=1e-2,default_value=1e-5 , log=True),
                    CS.UniformFloatHyperparameter("learning_rate" , lower=1e-5 , upper=1e-1,default_value=1e-3 ,log=True) ,
                    CS.UniformFloatHyperparameter("l1_weight",lower=1e-7,upper=1e-2,default_value=1e-5 , log=True),
                    CS.UniformFloatHyperparameter("batch_percentage",lower=0.05,upper=0.5, default_value=0.05),
                    CS.UniformFloatHyperparameter("p_drop",lower=0,upper=0.5, default_value=0.05),
                    CS.CategoricalHyperparameter("ODL", choices = [True, False], default_value=True),
                    CS.CategoricalHyperparameter("HDL", choices = [True, False] , default_value=True),
                    CS.Constant("boost_lr",value= 1e-3),
                    ]

    wraper_mape = {
        "LEdMLP_DLHO_Boost" : layer_wise_optimization, 
        "BLEdMLP_DLHO_Boost" : box_nodes_layer_wise_optimization
     }

    if args.dataset is None :
        datasets = ["abalone" , "arrhythmia" , "cardiotocography-10clases" , "cardiotocography-3clases" , "chess-krvkp"  , "congressional-voting" , "contrac" , "glass" , "molec-biol-splice" , "monks-3" , "musk-2" ,"oocytes_trisopterus_states_5b" , "spambase" , "statlog-image" , "statlog-landsat" ,"wall-following" , "waveform" , "waveform-noise", "breast-cancer-wisc-prog" , "breast-tissue" , "conn-bench-sonar-mines-rocks" , "conn-bench-vowel-deterding" , "hill-valley" , "ionosphere" , "iris" , "oocytes_merluccius_nucleus_4d" , "oocytes_merluccius_states_2f" , "oocytes_trisopterus_nucleus_2f" , "oocytes_trisopterus_states_5b" , "parkinsons" , "plant-shape" , "ringnorm" ,  "seeds" , "synthetic-control" , "twonorm" , "vertebral-column-2clases" , "vertebral-column-3clases"]
    else : 
        datasets = args.dataset

    if all([model in supported_models for model in args.model]) :
        models = args.model
    else :
        Exception(f"Model {args.model} not supported, supported models are {supported_models}")
    
    n_trials = args.n_trials
    random_state= args.random_state

    layer_patience = args.layer_patience
    epochs = args.epochs
    max_layers = args.max_layers

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
    for idx,  dataset_name in enumerate(datasets) : 
        if args.device == 'cuda' :
            device = f"cuda:{idx%2}"
        else : 
            device = args.device
        pool_requests[dataset_name] = []
        for model_name in models :
            try : 
                pool_res = pool.apply_async(wraper_mape[model_name], (dataset_name , model_name , bounds , device, run_id, n_trials , epochs , max_layers ,layer_patience, args.tunning_reps , args.eval_reps, random_state ) )
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
