# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:44:33 2022

@author: Hussein Aly
"""
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score
from DLoader.uci import UCIDataset
import glob
import pickle as pkl 
import datetime 
import pandas as pd 
import time
import random
from sklearn.model_selection import RepeatedKFold
from Models.backprob import MLP_model , EDBP
import os 
from Models.LWRVFL import LWTEDBP
from torch.multiprocessing import Pool, set_start_method
# util functions 
from Models.edRVFL import  BP_WPRVFL
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

def get_most_recent_run_id(hyper_tune_loc):
    
    recent_params = glob.glob(f"{hyper_tune_loc}/best*")

    if len(recent_params) > 1 :
        # choose most recent
        params_time = []
        for param in recent_params : 
            time = param.split('best_')[1].split('.')[0]
            params_time.append(datetime.datetime.strptime(time , "%d_%m_%YT_%H_%M"))
        recent_param_name = recent_params[np.argmax(params_time)]
        run_id = recent_param_name.split('best_')[1].split('.')[0]
    else :
        recent_param_name = recent_params[0]
        run_id = recent_param_name.split('best_')[1].split('.')[0]

    return recent_param_name , run_id

def get_param_by_id(dataset_name , model_name , id ) : 

    with open(f"hyperparam_tunning/{dataset_name}/{model_name}/best_{id}.pk" , "rb") as file :
        params = pkl.load(file)
    return params

def get_run_ids(datasets, models):
    
    results = {}

    previous_run_id = None 
    consistant = True 
    for dataset_name in datasets : 
        results[dataset_name] = {}
        for model_name in models : 
            param_name, run_id = get_most_recent_run_id(f"hyperparam_tunning/{dataset_name}/{model_name}/")
            results[dataset_name][model_name] = run_id
            if previous_run_id is None : 
                previous_run_id = run_id 
            else : 
                if run_id != previous_run_id : 
                    consistant = False 

    return results , consistant

# evaluators 

def EdMLP_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = False , boost = False , HDL=False , OHL=False, output_loc =check_point_loc , snn = False)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdSNN_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = False, boost = False , HDL=False , OHL=False, output_loc =check_point_loc , snn = True)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdMLP_Boost_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = params['boost_lr'] , HDL=False , OHL=False, output_loc =check_point_loc , snn = False)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdSNN_Boost_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = params['boost_lr'], HDL=False , OHL=False, output_loc =check_point_loc , snn = True)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdMLP_DLHO_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = False , boost = False , HDL=True , OHL=True, output_loc =check_point_loc , snn = False)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdSNN_DLHO_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = False , boost = False , HDL=True , OHL=True, output_loc =check_point_loc , snn = True)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdMLP_DLHO_EW_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = False , HDL=True , OHL=True, output_loc =check_point_loc , snn = False)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdSNN_DLHO_EW_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = False , HDL=True , OHL=True, output_loc =check_point_loc , snn = True)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdMLP_DLHO_Boost_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = params['boost_lr'], HDL=True , OHL=True, output_loc =check_point_loc , snn = False)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def EdSNN_DLHO_Boost_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = EDBP(classes=y_train.shape[1], nodes = params['nodes']  ,
                    layers=params['layers'] , epochs = 100,
                    learning_rate = params['learning_rate'],
                    weight_decay = params['weight_decay'],
                    l1_weight = params['l1_weight'], 
                    batch_percentage = params['batch_percentage'], device=device,
                    gamma = 1 , beta=0, p_drop = params['p_drop'],
                    bn_learnable=True, track_running_stats = True, seed = seed ,
                    verbose = 1, est_weight = True , boost = True , bootstrap = False, boost_lr = params['boost_lr'] , HDL=True , OHL=True, output_loc =check_point_loc , snn = True)


        best_epochs, estimator_weights = model.train(X=X_train,
                    y=y_train,
                    X_val = X_val,
                    y_val = y_val)
        
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=X, y=y , epochs=best_epochs )
        # model.set_estimator_weights(estimator_weights)

        return model 

def MLP_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = MLP_model(classes=y_train.shape[1], nodes = params['nodes'],
                        layers = params['layers'], epochs = 100,
                        learning_rate = params['learning_rate'],  l1_weight = params['l1_weight'],
                        device = device, batch_percentage = params['batch_percentage'],
                        weight_decay = params['weight_decay'] , direct_link = False ,
                        seed = seed, verbose = 1 , output_loc = check_point_loc , p_drop = params['p_drop'] , snn = False)

        best_epochs = model.train(X=torch.tensor(X_train).float(),
                                y=torch.tensor(y_train).float(), 
                                X_val= torch.tensor(X_val).float(), 
                                y_val= torch.tensor(y_val).float(), )
        # # combine train and val
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=torch.tensor(X).float(),
        #             y=torch.tensor(y).float(),  
        #             epochs=best_epochs)
        
        return model

def SNN_creator(X_train , y_train , X_val , y_val , params , seed , check_point_loc , device):
        model = MLP_model(classes=y_train.shape[1], nodes = params['nodes'],
                        layers = params['layers'], epochs = 100,
                        learning_rate = params['learning_rate'],  l1_weight = params['l1_weight'],
                        device = device, batch_percentage = params['batch_percentage'],
                        weight_decay = params['weight_decay'] , direct_link = False ,
                        seed = seed, verbose = 1 , output_loc = check_point_loc , p_drop = params['p_drop'] , snn = True)

        best_epochs = model.train(X=torch.tensor(X_train).float(),
                                y=torch.tensor(y_train).float(), 
                                X_val= torch.tensor(X_val).float(), 
                                y_val= torch.tensor(y_val).float(), )
        # # combine train and val
        # X = np.concatenate((X_train , X_val) , axis=0)
        # y = np.concatenate((y_train , y_val) , axis=0)

        # model.train(X=torch.tensor(X).float(),
        #             y=torch.tensor(y).float(),  
        #             epochs=best_epochs)
        return model 

def P_LWTEDBP_DLHO_Boost_creator(X_train , y_train , X_val , y_val , params, epochs, seed , check_point_loc , device):
   
    model =  LWTEDBP(classes=y_train.shape[1], epochs = epochs , gama = 1 , beta = 0 , bn_learnable=True , track_running_stats=True , verbose =2, est_weight = True ,  conf_weights = True ,boost = True , bootstrap = False, output_loc =check_point_loc , snn = False, device=device)

    train_weights = None
    val_weights = None

    for layer_id in range(len(params)) :
        _, _, _ , _ , layer_train_weights , layer_val_weights = model.train_layer(X=X_train, y=y_train, X_val = X_val, y_val = y_val, params = params[layer_id], train_sample_weight = train_weights , val_sample_weight = val_weights , append = True, seed=seed)

        train_weights = layer_train_weights
        val_weights = layer_val_weights       


    return model

def EdRVFL_creator(X_train , y_train , X_val , y_val , params, seed , check_point_loc , device) :

        bb_rvfl = BP_WPRVFL(classes=y_train.shape[1],
                        args={'C':params['C'], 'N':params['N'] } ,
                        layers=params['layers'] , epochs = 0,
                        learning_rate = 0 , low=1,
                        prune_rate = 0, weight_decay = 0,
                        batch_percentage = 1, gamma=params['gamma'],
                        beta=params['beta'],bn_affine=True,
                        device=device, seed = seed)
            

        train_pred , val_pred, layer_train_acc, layer_val_acc = bb_rvfl.train(X=torch.tensor(X_train).float(),
        y=torch.tensor(y_train).float(),
        X_val = torch.tensor(X_val).float(),
        y_val = torch.tensor(y_val).float())


        return bb_rvfl

def model_evaluator(dloader , params, model_func,  check_point_loc, reps=5, device="cuda", console=False , random_seed= 41, is_LW=False , epochs=100):

    metrics = []
    seeds = gen_ran_arr(reps , random_seed )

    cv_splits = dloader.n_CV
    for cv_idx in range(cv_splits) :
        X_train, y_train, X_val, y_val , X_test , y_test , _ , _ = dloader.getitem(cv_idx)
        for rep_idx in range(reps):
            torch.cuda.empty_cache()

            if is_LW : 
                model = model_func(X_train= X_train , y_train= y_train , X_val = X_val , y_val = y_val, params= params, epochs = epochs, seed = seeds[rep_idx], check_point_loc = check_point_loc , device=device)
            else : 
                model = model_func(X_train= X_train , y_train= y_train , X_val = X_val , y_val = y_val, params= params, seed = seeds[rep_idx], check_point_loc = check_point_loc , device=device)

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


model_func_map = {
    "SNN" : SNN_creator,
    "MLP" : MLP_creator,
    "EdMLP"  : EdMLP_creator,
    "EdSNN" : EdSNN_creator,
    "EdMLP_DLHO"  : EdMLP_DLHO_creator,
    "EdSNN_DLHO" : EdSNN_DLHO_creator,
    "EdMLP_DLHO_EW" : EdMLP_DLHO_EW_creator,
    "EdSNN_DLHO_EW" : EdSNN_DLHO_EW_creator, 
    "EdMLP_Boost"  : EdMLP_Boost_creator,
    "EdSNN_Boost" : EdSNN_Boost_creator,
    "EdMLP_DLHO_Boost"  : EdMLP_DLHO_Boost_creator,
    "EdSNN_DLHO_Boost" : EdSNN_DLHO_Boost_creator,
    "LEdMLP_DLHO_Boost" : P_LWTEDBP_DLHO_Boost_creator, 
    "BLEdMLP_DLHO_Boost" : P_LWTEDBP_DLHO_Boost_creator, 

    "EdRVFL" : EdRVFL_creator,

}

if __name__ == "__main__" : 
    supported_models = model_func_map.keys()
    parser = argparse.ArgumentParser(description='Model Evaluator Script')
    parser.add_argument('--run_id', type=str, default="auto", help='Run ID of the hyperparamter tunning, leave auto to use the most recent Id')
    parser.add_argument('--device', type=str, default="cpu", help='Device to run the model on')
    parser.add_argument('--model', type=str, nargs="+", required=True, choices=supported_models, help='Model to use')
    parser.add_argument('--dataset', type=str, nargs="+", help='Dataset to use')
    parser.add_argument('--random_seed', type=int, default=41, help='Random seed for the evaluation')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to run')
    parser.add_argument('--epochs', type=int, default=100, help='Training Epochs for used For Layer wise algorithms only, otherwise ignored.')
    parser.add_argument('--eval_reps', type=int, default=5, help='Number of repetitions for evaluation')

    args = parser.parse_args()

    if args.dataset is None :
        datasets = ["abalone" , "arrhythmia" , "cardiotocography-10clases" , "cardiotocography-3clases" , "chess-krvkp"  , "congressional-voting" , "contrac" , "glass" , "molec-biol-splice" , "monks-3" , "musk-2" ,"oocytes_trisopterus_states_5b" , "spambase" , "statlog-image" , "statlog-landsat" ,"wall-following" , "waveform" , "waveform-noise", "breast-cancer-wisc-prog" , "breast-tissue" , "conn-bench-sonar-mines-rocks" , "conn-bench-vowel-deterding" , "hill-valley" , "ionosphere" , "iris" , "oocytes_merluccius_nucleus_4d" , "oocytes_merluccius_states_2f" , "oocytes_trisopterus_nucleus_2f" , "oocytes_trisopterus_states_5b" , "parkinsons" , "plant-shape" , "ringnorm" ,  "seeds" , "synthetic-control" , "twonorm" , "vertebral-column-2clases" , "vertebral-column-3clases"]
    else : 
        datasets = args.dataset



    models = args.model
    device= args.device
    run_id = args.run_id
    random_seed = args.random_seed
    session_id = datetime.datetime.now().strftime("%d_%m_%YT_%H_%M")

    LW_models = ["BLEdMLP_DLHO_Boost" , "LEdMLP_DLHO_Boost"] 

    if run_id == "auto" : 
        run_ids , consistant = get_run_ids(datasets , [model_name])
        if consistant : 
            output_file = f"results_r{run_ids[datasets[0]][model_name]}_consistant_s{session_id}.csv"
        else : 
            output_file = f"results_not_consistant_s{session_id}.csv"
    else : 
        output_file = f"results_r{run_id}_consistant_s{session_id}.csv"
    
    
    print(f"Run ID : {run_id}")
    print(f"Session ID : {session_id}")

    pool = Pool(args.n_jobs)
    pool_requests = {}
    for idx,  dataset_name in enumerate(datasets) : 
        if device == "cuda" : 
            device = f"cuda:{idx%2}"
        else : 
            device = args.device
        pool_requests[dataset_name] = []
        loader = UCIDataset(dataset_name, parent="DLoader/UCIdata")

        if run_id == "auto" : 
            best_params = get_param_by_id(dataset_name, model_name , run_ids[dataset_name][model_name])
        else : 
            best_params = get_param_by_id(dataset_name, model_name , run_id)
                    
        check_point_loc = f"hyperparam_tunning/{dataset_name}/{model_name}/checkpoints/"

        if not os.path.exists(check_point_loc):
            os.makedirs(check_point_loc)

        try : 
            if model_name in LW_models :
                pool_res = pool.apply_async(model_evaluator, (loader, best_params['params'] , model_func_map[model_name],  check_point_loc, args.eval_reps, device, True , random_seed, True, args.epochs))
            else : 
                pool_res = pool.apply_async(model_evaluator, (loader, best_params['params'] , model_func_map[model_name],  check_point_loc, args.eval_reps, device, True , random_seed))

            pool_requests[dataset_name].append(pool_res)
        except Exception as e : 
            print(e)
    
    total_res = pd.DataFrame(columns=["train_acc",  "train_f1", "train_precision",  "train_recall",  "val_acc",  "val_f1", "val_precision",  "val_recall" , "test_acc", "test_f1", "test_precision", "test_recall", "model" , "dataset" ,  "duration"  ])
    
    for dataset_name in pool_requests.keys(): 
        for request in pool_requests[dataset_name] :
            try : 
                request.wait()
                result =  request.get()
                if not result is None :
                    train_acc,  train_f1, train_precision,  train_recall,  val_acc,  val_f1, val_precision,  val_recall , test_acc, test_f1, test_precision, test_recall = result

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
                    
                    total_res = pd.concat((total_res , results_df ) )
                    total_res.to_csv(f"results/{output_file}", index=False)
            except Exception as e : 
                print(e) 
    pool.close()
    pool.join()
