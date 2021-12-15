from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import csv
import itertools
import sys
#from random import randint
#from tqdm import tqdm

from sklearn import metrics
from scipy import stats

model_name='InterBKT'

ALMOST_ONE = 0.999
ALMOST_ZERO = 0.001

class BKT(BaseEstimator):
    def __init__(self, step = 0.1, bounded = True, best_k0 = True):
        self.k0 = ALMOST_ZERO
        self.transit = ALMOST_ZERO
        self.guess = ALMOST_ZERO
        self.slip = ALMOST_ZERO
        self.forget = ALMOST_ZERO

        self.k0_limit = ALMOST_ONE
        self.transit_limit = ALMOST_ONE
        self.guess_limit = ALMOST_ONE
        self.slip_limit = ALMOST_ONE
        self.forget_limit = ALMOST_ONE

        self.step = step
        self.best_k0 = best_k0

        if bounded:
            self.k0_limit = 0.85
            self.transit_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X, y = None):

        if self.best_k0:
            self.k0 = self._find_best_k0(X)
            self.k0_limit = self.k0

        k0s = np.arange(self.k0,
            min(self.k0_limit + self.step, ALMOST_ONE),
            self.step)
        transits = np.arange(self.transit,
            min(self.transit_limit + self.step, ALMOST_ONE),
            self.step)
        guesses = np.arange(self.guess,
            min(self.guess_limit + self.step, ALMOST_ONE),
            self.step)
        slips = np.arange(self.slip,
            min(self.slip_limit + self.step, ALMOST_ONE),
            self.step)

        all_parameters = [k0s, transits, guesses, slips]
        #print(len(all_parameters))
        parameter_pairs = list(itertools.product(*all_parameters))
        #print (len(parameter_pairs))

        min_error = sys.float_info.max
        #for (k, t, g, s) in tqdm(parameter_pairs, leave=False):
        #print(X)
        for (k, t, g, s) in (parameter_pairs):
            #print(k, t, g, s)
            error, _ = self._computer_error(X, k, t, g, s)
            if error < min_error:
                self.k0 = k
                self.transit = t
                self.guess = g
                self.slip = s
                min_error = error

        #print "Traning RMSE: ", min_error
        #print 
        return self.k0, self.transit, self.guess, self.slip

    def _computer_error(self, X, k, t, g, s):
        error = 0.0
        n = 0
        predictions = []

        for seq in X:            
            current_pred = []
            pred = k
            for i, res in enumerate(seq):
                n += 1
                current_pred.append(pred)
                error += (res - pred) ** 2
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = p + (1 - p) * t
                pred = k * (1 - s) + (1 - k) * g
                # pred = k
            predictions.append(current_pred)

        return (error / n)  ** 0.5, predictions

    def _find_best_k0(self, X):
        res=0.5
        kc_best= np.mean([seq[0] for seq in X])
        if  kc_best>0:
            res=kc_best        
                           
        return res

    def predict(self, X, L, T, G, S):
        return self._computer_error(X, L, T, G, S)
        

      
    def inter_predict(self, S, X, k, t, g, s, num_skills):
        error = 0.0
        n = 0
        all_all_mastery = {}   
        
        for j in S.keys(): 
            skills=list(map(int,S[j]))
            responses=list(map(int,X[j]))           
            last_mastery=np.zeros(num_skills)
        
            if len(skills)>1:
             
               ini_skill=[]
               all_mastery = []
               pL = np.zeros(len(skills)+1)
               
               
               for i in range(len(skills)):    
                   
                   if i<len(skills):
                   
                      skill_id= skills[i]
                      
                      
                      if skill_id not in ini_skill:
                         ini_skill.append(skill_id)
                         pL[i]=k[skill_id]
                      else:
                          pL[i]=last_mastery[skill_id]
                          
                      mastery = pL[i]    # mastery is assess before updating with response
                      all_mastery.append(mastery)
                      
                      
                      
                      
                      res=responses[i]    # update the mastery when response is known
                      if res == 1.0:  
                         pL[i+1] = pL[i] * (1 - s[skill_id]) / (pL[i] * (1 - s[skill_id]) + (1 - pL[i]) * g[skill_id])
                      else:
                           pL[i+1] = pL[i] * s[skill_id] / (pL[i] * s[skill_id] + (1 - pL[i]) * (1 - g[skill_id]))
                      pL[i+1] = pL[i+1] + (1 - pL[i+1]) * t[skill_id]
                      last_mastery[skill_id]=pL[i+1]
               
               all_all_mastery[j]=all_mastery
               
               
               
        
        return all_all_mastery
    #print (predictions)

if __name__ == "__main__":
   main()

