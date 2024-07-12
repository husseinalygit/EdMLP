

# write a class for creating a model based on adaboosting, using the same interface as EDBP model 
# Path: SNN\Models\boosting.py

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

class AdaBoosting :
    def __init__(self , n_estimators = 50 , learning_rate = 0.1 , random_state = 42 ) : 
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = AdaBoostClassifier(n_estimators = n_estimators , learning_rate = learning_rate , random_state = random_state)

    def train(self , X , y) : 
        self.model.fit(X , y.argmax(1))
        return self

    def predict(self , X) : 
        return self.model.predict_proba(X)

