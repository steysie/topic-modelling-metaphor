from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import ast, csv
from statistics import mean
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Function to compute delta P
def compute_statistics(contingency_smoothed, met_corpus_size, nonmet_corpus_size):
   
    dp1={}

    for line in contingency_smoothed:
        word = line[0]
        a = line[1]
        b = line[2]
        c = met_corpus_size - a
        d = nonmet_corpus_size - b

        res = a/(a+b) - c/(c+d)
       
        dp1[word] = res

    return dp1

# Function to vectorize lemmas in the windows: assign the statistic score to every word; 
def assign_scores(text, statistic_scores):

    means = []
    
    for line in text:
        line_score = []
        if line != []:
            for word in line:
                if word in statistic_scores:
                    line_score.append(statistic_scores.get(word))
                else:
                    line_score.append(float(0))
        else:            
            line_score.append(float(0))
            
        line_score_mean = np.array(line_score).mean()
        means.append(line_score_mean)

    return pd.DataFrame(means)

# Function for calculating cooccurrence matrices
def freq_table(array, verb_dict):
    lemmas_met = []
    lemmas_nonmet = []
    met_freqs = []
    nonmet_freqs = []

    for pair in array:
        cl = pair[0]
        text = pair[1]
        for word in text:
            if word not in verb_dict and word != 'который':
                if cl == 1:
                    lemmas_met.append(word)
                else:
                    lemmas_nonmet.append(word)

#     print('table1', len(lemmas_met), len(lemmas_nonmet))
                    
    met_lemmas_count = Counter(lemmas_met)
    nonmet_lemmas_count = Counter(lemmas_nonmet)
    
                    
    for k, v in met_lemmas_count.items():
        met_freqs.append(v)
        
    for k, v in nonmet_lemmas_count.items():
        nonmet_freqs.append(v)
        
    met_corpus_size = sum(met_freqs) + 1 
    nonmet_corpus_size = sum(nonmet_freqs) + 1 
    
    
    contingency_table = []
    
        
    for key in set(list(met_lemmas_count.keys()) + list(nonmet_lemmas_count.keys())):
        contingency_table.append(
            [
                key, 
                met_lemmas_count[key] if key in met_lemmas_count.keys() else 0, 
                nonmet_lemmas_count[key] if key in nonmet_lemmas_count.keys() else 0
            ]
        )
        
        
    contingency_smoothed = []   
    
    mets_smooth = 0
    nonmets_smooth = 0
    
    for pair in contingency_table:
        word = pair[0]
        a = pair[1]
        b = pair[2]

        if a == 0:
            mets_smooth +=1 
        if b == 0:
            nonmets_smooth +=1
            
 
        
    for pair in contingency_table:
        word = pair[0]
        a = pair[1]
        b = pair[2]   
        
        if a == 0:
            n_a = 1 / mets_smooth
        else:
            n_a = float(a)
        if b == 0:
            n_b = 1 / nonmets_smooth
        else:
            n_b = float(b)

        contingency_smoothed.append([word, n_a, n_b])

    result = [contingency_smoothed, met_corpus_size, nonmet_corpus_size] 

    return result