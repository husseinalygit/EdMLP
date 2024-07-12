# -*- coding: utf-8 -*-
import numpy as np
from easydict import EasyDict
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class WPPreceptron(nn.Module):
    def __init__(self, features, raw_features, classes , nodes, lamb, device, prune_rate, gamma, beta, bn_affine):
        super(WPPreceptron, self).__init__()

        # init params
        self.features = features
        self.raw_features = raw_features
        self.classes = classes
        self.nodes = nodes
        self.lamb = lamb
        self.d = device
        self.n_prune = int(np.floor(prune_rate * nodes))
        self.prune_matrix = torch.ones(self.nodes).to(self.d)
        self.gamma = gamma
        self.beta = beta
        self.bn_affine = bn_affine

        # init layers
        self.layer = nn.Sequential(
            nn.Linear(self.features , self.nodes),
            nn.ReLU(),
            nn.BatchNorm1d(self.nodes, affine=beta, track_running_stats=True),
            # nn.InstanceNorm1d(self.nodes),
            # nn.Dropout1d(0.05),
        )
        self.layer.apply(self.__init_weights__)

        self.output = nn.Sequential(
                nn.Linear(self.nodes + self.raw_features +1, classes),
                nn.Softmax(dim=1))

    def __init_weights__(self, m):
        if isinstance(m, nn.Linear):
            # y = m.in_features
            # m.weight.data.normal_(0.0,1/np.sqrt(y))
            # m.bias.data.fill_(0)
            m.weight.data.uniform_(-1 , 1)
            m.bias.data.uniform_(0, 1)
        elif isinstance(m, nn.BatchNorm1d) :
            m.weight.data.fill_(self.gamma)
            m.bias.data.fill_(self.beta)
        

    def init_weight(self, X , y, X_raw, weighted_matrix):

        n_sample, n_features = X.shape
        encoding = self.transform(X)
        merged = torch.cat([X_raw, encoding, torch.ones((n_sample, 1)).to(self.d)], axis=1)
        W = torch.zeros(X.shape[0] , X.shape[0]).to(self.d)
        W[range(len(W)), range(len(W))] = weighted_matrix

        if n_features<n_sample:
            beta = torch.mm(torch.mm(torch.inverse(torch.eye(merged.shape[1]).to(self.d)/self.lamb+torch.mm(torch.mm(merged.T , W),merged)),torch.mm(merged.T , W)),y)
        else:
            beta = torch.mm(merged.T,torch.mm(torch.linalg.inv(torch.eye(merged.shape[0]).to(self.d)/self.lamb+torch.mm(torch.mm(W , merged),merged.T)),torch.mm(W, y)))

        # pruning
        beta_sum = torch.sum(torch.abs(beta[beta.shape[0]- self.nodes : , : ]) , 1)
        beta_sorted_idx = torch.argsort(beta_sum)

        self.prune_matrix[beta_sorted_idx[:self.n_prune]] = 0

        # print(len(self.output[0].weight[0]))
        self.output[0].weight = nn.Parameter(beta.T)
        self.output[0].bias.data.fill_(0)
        # print(len(self.output[0].weight[0]))



    def forward(self, X, X_raw) :
        encoding = self.layer(X)
        encoding = encoding * self.prune_matrix.repeat(X.shape[0], 1)
        merged = torch.cat([X_raw, encoding, torch.ones((X_raw.shape[0], 1)).to(self.d)], axis=1)
        probability = self.output(merged)
        return probability

    def transform(self, X):
        encoding = self.layer(X)
        encoding = encoding * self.prune_matrix.repeat(X.shape[0], 1)
        return encoding

class BP_WPRVFL():

    # BP version based on forward method
    def __init__(self, classes, args, layers= 5 , epochs = 10, learning_rate = 1e-3,
                 low=1, prune_rate = 0, device='cpu',weight_decay =0, batch_percentage= 0.05,
                 gamma = 1 , beta = 0, bn_affine = True,
                 l1_weight= 0, rvfl_first = True, cfs = True , seed = -1):

        super().__init__()
        self.args = EasyDict(args)
        self.args.lamb = 2**self.args.C
        self.classes = classes
        self.Params = EasyDict()
        self.d = device
        self.state_init()
        self.layers = layers
        self.epochs = epochs
        self.lr = learning_rate
        self.low = low
        self.prune_rate = prune_rate
        self.weight_decay = weight_decay
        self.batch_percentage = batch_percentage
        self.gamma = gamma
        self.beta = beta
        self.bn_affine = bn_affine
        self.l1_weight = l1_weight
        self.rvfl_first = rvfl_first
        self.seed = seed
        self.cfs = cfs
        # np.random.seed(32)
        # torch.manual_seed(32)

    def state_init(self):
        self.Params=EasyDict()
        self.Params['train_pred'] = []
        self.Params['val_pred'] = []
        self.Params['train_acc'] = []
        self.Params['val_acc'] = []
        self.Params["epochs_trian_scores"] = []
        self.Params["epochs_val_scores"] = []
        self.Params["models"] = []


    def train(self, X , y, X_val , y_val):

        if self.seed != -1 :
            self.fix_seed(self.seed)


        X = torch.Tensor(X).float().to(self.d)
        y = torch.Tensor(y).float().to(self.d)
        X_val = torch.Tensor(X_val).float().to(self.d)
        y_val = torch.Tensor(y_val).float().to(self.d)

        train_encoding = X.clone().to(self.d)
        val_encoding = X_val.clone().to(self.d)

        weighted_matrix = torch.ones(X.shape[0]).to(self.d)
        n_steps = np.ceil(1 / self.batch_percentage)


        # compute class weihgts

        # class_weights = compute_class_weight("balanced" , classes= np.unique(y.clone().cpu().detach().numpy()) , y=y.clone().cpu().detach().numpy().argmax(1))

        for layer_id in range(int(self.layers)) :

            model = WPPreceptron(train_encoding.shape[1] , X.shape[1] ,
                               y.shape[1] , int(self.args.N),
                               self.args.lamb, self.d , self.prune_rate, self.gamma, self.beta, self.bn_affine).to(self.d)

            if self.rvfl_first & self.cfs:
                model.train()
                model.init_weight(train_encoding,  y , X, weighted_matrix)

            # print(f"temp_enc mean {temp_enc.mean() : 0.2f},std {temp_enc.std() : 0.2f} ")

            # start back prop
            # torch.Tensor(class_weights).to(self.d)
            loss_fn = nn.CrossEntropyLoss()
            # loss_fn = nn.MSELoss()
            # print("CE-Adam")
            opt = torch.optim.Adam(model.parameters() , lr = self.lr, weight_decay=self.weight_decay)

            layer_train_pred = []
            layer_train_acc = []
            layer_val_pred = []
            layer_val_acc = []

            # print(f"starting BP layer {layer_id}")
            for epoch in range(int(self.epochs)) :
                # model.train()
                #
                # train_pred = model(train_encoding, X)
                # loss = loss_fn(train_pred , y)
                # opt.zero_grad()
                # loss.backward()
                # opt.step()
                self.train_loop(train_encoding,  X, y,
                                model , loss_fn , opt)
                # print(f"Layer weights mean {model.layer[0].weight.mean():0.2f}, std {model.layer[0].weight.std():0.2f}, sum {model.layer[0].weight.sum():0.2f}")

                model.eval()
                with torch.no_grad():
                    train_pred = model(train_encoding, X)
                    loss = loss_fn(train_pred , y)
                    train_pred = train_pred.cpu().detach().numpy()
                    train_acc = accuracy_score(y.argmax(1).cpu() , train_pred.argmax(1))

                    val_pred = model(val_encoding, X_val)
                    val_pred = val_pred.cpu().detach().numpy()
                    val_acc = accuracy_score(y_val.cpu().argmax(1) , val_pred.argmax(1))
                    # print(f"Epoch {epoch}: loss {loss} using {n_steps} step/s : train_acc {train_acc*100:0.2f} : val_acc {val_acc * 100 :0.2f}")
                    # layer_train_pred.append(train_pred)
                    layer_train_acc.append(train_acc)
                    # layer_val_pred.append(val_pred)
                    layer_val_acc.append(val_acc)

            if (not self.rvfl_first) & self.cfs :
                model.init_weight(train_encoding,  y , X, weighted_matrix)
            model.eval()
            with torch.no_grad():
                # predict on full x to calculate encoding and weight matrix.
                train_pred = model(train_encoding, X)
                train_pred = train_pred.cpu().detach().numpy()

                val_pred = model(val_encoding, X_val)
                val_pred = val_pred.cpu().detach().numpy()



            self.Params['train_pred'].append(train_pred)
            self.Params['val_pred'].append(val_pred)

            self.Params['train_acc'].append(layer_train_acc)
            self.Params['val_acc'].append(layer_val_acc)


            # weighting
            train_labels = train_pred.argmax(1)
            wrong_idx = np.where(train_labels != y.argmax(1).cpu().numpy() )[0]
            correct_idx = np.where(train_labels  == y.argmax(1).cpu().numpy())[0]

            n_correct = len(correct_idx)
            n_wrong = len(wrong_idx)

            if n_wrong == 0 :
                high = 0
            else :
                high = (X.shape[0] - n_correct * self.low)/n_wrong

            weighted_matrix[wrong_idx] = high
            weighted_matrix[correct_idx] = self.low


            self.Params['models'].append(model)

            train_encoding = model.transform(train_encoding).detach().clone()
            val_encoding = model.transform(val_encoding)

        return  self.Params['train_pred'] , self.Params['val_pred'], self.Params['train_acc'] ,  self.Params['val_acc']

    def fix_seed(self , seed):
        torch.manual_seed(seed)
        np.random.seed(seed)


    def train_loop(self , encoding , X , y, model , loss_fn , opt) :
        model.train()

        batch_size =  int(np.ceil(self.batch_percentage * X.shape[0]))
        indexs = torch.randperm(X.shape[0]) # premutated to have random shuffling

        for batch_id in range(0 , X.shape[0] , batch_size) :
            batch_idx = indexs[batch_id : batch_id + batch_size ]
            train_pred = model(encoding[batch_idx], X[batch_idx])
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
        X = torch.Tensor(X).to(self.d)
        scores = []
        encoding = X.clone().to(self.d)
        for i , model in enumerate(self.Params.models) :
            model.eval()
            pred_score = model(encoding, X)
            pred_score = pred_score.cpu().detach().numpy()
            scores.append(pred_score)
            encoding = model.transform(encoding)
        scores = np.array(scores)
        return scores.mean(0)