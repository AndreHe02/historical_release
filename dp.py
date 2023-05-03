import numpy as np
from data import vocab
import torch 
from models import ProbCache

BIG_NEG = -1e9

def compute_mutation_prob(model, sources, targets, return_posteriors=False):
    raw_sources, raw_targets = sources, targets
    sources = vocab.make_tensor(sources, add_boundaries=True).numpy()   # faster in numpy
    targets = vocab.make_tensor(targets, add_boundaries=True).numpy()
    batch_size = sources.shape[1]
    
    model.cache_probs(sources, targets)

    f_sub = np.full((len(sources), len(targets), batch_size), fill_value=BIG_NEG)
    f_ins = np.full((len(sources), len(targets), batch_size), fill_value=BIG_NEG)
    # start by subbing ( with (. Now in insertion mode
    f_ins[0, 0] = 0.0 
    
    for i in range(len(sources)):   # start at i=0, might insert after (
        for j in range(len(targets)):
            
            if j > 0:
                f_ins[i, j] = np.logaddexp(
                    f_ins[i, j-1] + model.ins(i, j-1),  # inserted last char
                    f_sub[i, j-1] + model.sub(i, j-1)   # subbed last char
                )
            
            if i > 0:
                f_sub[i, j] = np.logaddexp(
                    f_ins[i-1, j] + model.end(i-1, j),  # just ended insertion
                    f_sub[i-1, j] + model.dlt(i-1, j)   # deleted last char
                )
    
    source_final_idx = [len(s)+1 for s in raw_sources]
    target_final_idx = [len(s) for s in raw_targets]
    total_pr = f_sub[source_final_idx, target_final_idx, range(batch_size)]

    if not return_posteriors:
        return total_pr 
        
    f_sub_post = np.full((len(sources), len(targets), batch_size), fill_value=BIG_NEG)
    f_ins_post = np.full((len(sources), len(targets), batch_size), fill_value=BIG_NEG)
    f_sub_post[source_final_idx, target_final_idx, range(batch_size)] = 0.0

    posterior_cache = ProbCache(sources, targets)

    for i in range(len(sources)-1, -1, -1):
        for j in range(len(targets)-1, -1, -1):

            if j < len(targets)-1:
                ins_post_pr = (f_ins[i, j] + model.ins(i, j) - f_ins[i, j+1]) + f_ins_post[i, j+1]
                f_ins_post[i, j] = np.logaddexp(f_ins_post[i, j], ins_post_pr)
                posterior_cache.ins[i, j] = ins_post_pr 

                sub_post_pr = (f_sub[i, j] + model.sub(i, j) - f_ins[i, j+1]) + f_ins_post[i, j+1] 
                f_sub_post[i, j] = np.logaddexp(f_sub_post[i, j], sub_post_pr)
                posterior_cache.sub[i, j] = sub_post_pr

            if i < len(sources)-1:
                dlt_post_pr = (f_sub[i, j] + model.dlt(i, j) - f_sub[i+1, j]) + f_sub_post[i+1, j]
                f_sub_post[i, j] = np.logaddexp(f_sub_post[i, j], dlt_post_pr)
                posterior_cache.dlt[i, j] = dlt_post_pr

                end_post_pr = (f_ins[i, j] + model.end(i, j) - f_sub[i+1, j]) + f_sub_post[i+1, j]
                f_ins_post[i, j] = np.logaddexp(f_ins_post[i, j], end_post_pr)
                posterior_cache.end[i, j] = end_post_pr

    return posterior_cache

def compute_posteriors(model, sources, targets):
    return compute_mutation_prob(model, sources, targets, return_posteriors=True)
