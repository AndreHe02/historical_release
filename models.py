import numpy as np 
import torch 
from data import vocab 

"""
Indexing convention: [i, j] represents the context where:
- Input char is x_i (decoding from x_i)
- Just generated y_j 
if in substitution mode, about to:
    -substitute x_i with y_{j+1} OR
    -delete x_i
if in insertion mode, about to:
    -insert y_{j+1} OR
    -end the insertion event (move on to x_{i+1})

Models are allowed to condition on the entirety of x and y_{:j+1}
"""

BIG_NEG = -1e9


class ProbCache:
    def __init__(self, sources, targets, fill=BIG_NEG):
        batch_size = sources.shape[1]
        self.sub = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.dlt = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.ins = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.end = np.full((len(sources), len(targets), batch_size), fill_value=fill)

class TensorProbCache:
    def __init__(self, sources, targets, fill=BIG_NEG):
        batch_size = sources.shape[1]
        self.sub = torch.full((len(sources), len(targets), batch_size), fill_value=fill).cuda()
        self.dlt = torch.full((len(sources), len(targets), batch_size), fill_value=fill).cuda()
        self.ins = torch.full((len(sources), len(targets), batch_size), fill_value=fill).cuda()
        self.end = torch.full((len(sources), len(targets), batch_size), fill_value=fill).cuda()



class BaseModel:
    
    def sub(self, source, targets, i, j):
        return NotImplemented

    def dlt(self, source, targets, i, j):
        return NotImplemented

    def ins(self, source, targets, i, j):
        return NotImplemented

    def end(self, source, targets, i, j):
        return NotImplemented
    
    def cache_probs(self, sources, targets):
        return NotImplemented

    def m_step(self, sources, targets, posterior_cache):
        return NotImplemented


class DirichletModel(BaseModel):

    def __init__(self, initial_prs, natural_classes):
        """
        num_cls: number of natural classes 
        cls_of[id(x)] should give class id of character x 
        """
        self_sub, dlt, end_ins = initial_prs
        self.num_cls = max(natural_classes) + 1
        self.cls_of = natural_classes # np.zeros(vocab.size, dtype=np.int32)
        
        sub_pr = (1.0 - dlt - self_sub) / vocab.size # roughly
        self.sub_prs = np.full((self.num_cls, vocab.size, vocab.size), fill_value=sub_pr)
        self.sub_prs[:, range(vocab.size), range(vocab.size)] = self_sub
        self.sub_prs[:, :, vocab.step] = dlt
        
        ins_pr = (1.0 - end_ins) / vocab.size
        self.ins_prs = np.full((self.num_cls, vocab.size, vocab.size), fill_value=ins_pr)
        self.ins_prs[:, :, vocab.step] = end_ins

        self.normalize()

        self.init_sub_prs = np.exp(self.sub_prs)
        self.init_ins_prs = np.exp(self.ins_prs)    # keep for smoothing later 
        
        self.aligner = self 

    def normalize(self):
        self.sub_prs = self.sub_prs / self.sub_prs.sum(axis=-1, keepdims=True)
        self.sub_prs = np.log(self.sub_prs)
        self.ins_prs = self.ins_prs / self.ins_prs.sum(axis=-1, keepdims=True)
        self.ins_prs = np.log(self.ins_prs)

    def __sub_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]] 
        return self.sub_prs[left_context, input_char, targets[j+1]]

    def __dlt_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]] 
        return self.sub_prs[left_context, input_char, vocab.dlt]
    
    def __ins_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]]
        return self.ins_prs[left_context, input_char, targets[j+1]] 

    def __end_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]]
        return self.ins_prs[left_context, input_char, vocab.dlt] 
                
    
    def cache_probs(self, sources, targets):
        batch_size = sources.shape[1]
        self.cache = ProbCache(sources, targets)

        for i in range(len(sources)):
            for j in range(len(targets)):

                if j+1 < len(targets):
                    self.cache.sub[i, j] = self.__sub_pr(sources, targets, i, j) 
                    self.cache.ins[i, j] = self.__ins_pr(sources, targets, i, j)
                
                self.cache.dlt[i, j] = self.__dlt_pr(sources, targets, i, j)
                self.cache.end[i, j] = self.__end_pr(sources, targets, i, j)
                
    def sub(self, i, j):
        return self.cache.sub[i, j]
    
    def ins(self, i, j):
        return self.cache.ins[i, j]

    def dlt(self, i, j):
        return self.cache.dlt[i, j]

    def end(self, i, j):
        return self.cache.end[i, j]

    def m_step(self, sources, targets, posterior_cache, smooth=1.0):
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy() 
        batch_size = sources.shape[1]
        self.sub_prs = np.zeros((self.num_cls, vocab.size, vocab.size))
        self.ins_prs = np.zeros((self.num_cls, vocab.size, vocab.size))

        sub_post = np.exp(posterior_cache.sub)
        dlt_post = np.exp(posterior_cache.dlt)
        ins_post = np.exp(posterior_cache.ins)
        end_post = np.exp(posterior_cache.end)

        for i in range(len(sources)):
            for j in range(len(targets)):
                input_char = sources[i]
                left_context = self.cls_of[targets[j]]

                if j < len(targets)-1:
                    outcomes = targets[j+1] 
                    for n in range(batch_size): # have to loop cuz potentially repeating indices
                        self.sub_prs[left_context[n], input_char[n], outcomes[n]] += sub_post[i, j, n]
                        self.ins_prs[left_context[n], input_char[n], outcomes[n]] += ins_post[i, j, n]

                for n in range(batch_size):
                    self.sub_prs[left_context[n], input_char[n], vocab.dlt] += dlt_post[i, j, n]
                    self.ins_prs[left_context[n], input_char[n], vocab.dlt] += end_post[i, j, n]

        self.sub_prs = self.sub_prs + smooth * self.init_sub_prs
        self.ins_prs = self.ins_prs + smooth * self.init_ins_prs
        self.normalize()


class MarkednessModel(DirichletModel):
    """
    Instead of conditioning on last target char,
    condition on last source char. 
    It appears that conditioning on last target char does not do much.
    """

    def __sub_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[sources[i-1] if i>0 else [vocab.sow]*sources.shape[1]] 
        return self.sub_prs[left_context, input_char, targets[j+1]]

    def __dlt_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[sources[i-1] if i>0 else [vocab.sow]*sources.shape[1]] 
        return self.sub_prs[left_context, input_char, vocab.dlt]
    
    def __ins_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[sources[i-1] if i>0 else [vocab.sow]*sources.shape[1]] 
        return self.ins_prs[left_context, input_char, targets[j+1]] 

    def __end_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[sources[i-1] if i>0 else [vocab.sow]*sources.shape[1]] 
        return self.ins_prs[left_context, input_char, vocab.dlt] 
                
    
    def cache_probs(self, sources, targets):
        batch_size = sources.shape[1]
        self.cache = ProbCache(sources, targets)

        for i in range(len(sources)):
            for j in range(len(targets)):

                if j+1 < len(targets):
                    self.cache.sub[i, j] = self.__sub_pr(sources, targets, i, j) 
                    self.cache.ins[i, j] = self.__ins_pr(sources, targets, i, j)
                
                self.cache.dlt[i, j] = self.__dlt_pr(sources, targets, i, j)
                self.cache.end[i, j] = self.__end_pr(sources, targets, i, j)
                
    def sub(self, i, j):
        return self.cache.sub[i, j]
    
    def ins(self, i, j):
        return self.cache.ins[i, j]

    def dlt(self, i, j):
        return self.cache.dlt[i, j]

    def end(self, i, j):
        return self.cache.end[i, j]

    def m_step(self, sources, targets, posterior_cache, smooth=1.0):
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy() 
        batch_size = sources.shape[1]
        self.sub_prs = np.zeros((self.num_cls, vocab.size, vocab.size))
        self.ins_prs = np.zeros((self.num_cls, vocab.size, vocab.size))

        sub_post = np.exp(posterior_cache.sub)
        dlt_post = np.exp(posterior_cache.dlt)
        ins_post = np.exp(posterior_cache.ins)
        end_post = np.exp(posterior_cache.end)

        for i in range(len(sources)):
            for j in range(len(targets)):
                input_char = sources[i]
                left_context = self.cls_of[sources[i-1] if i>0 else [vocab.sow]*batch_size] 

                if j < len(targets)-1:
                    outcomes = targets[j+1] 
                    for n in range(batch_size): # have to loop cuz potentially repeating indices
                        self.sub_prs[left_context[n], input_char[n], outcomes[n]] += sub_post[i, j, n]
                        self.ins_prs[left_context[n], input_char[n], outcomes[n]] += ins_post[i, j, n]

                for n in range(batch_size):
                    self.sub_prs[left_context[n], input_char[n], vocab.dlt] += dlt_post[i, j, n]
                    self.ins_prs[left_context[n], input_char[n], vocab.dlt] += end_post[i, j, n]

        self.sub_prs = self.sub_prs + smooth * self.init_sub_prs
        self.ins_prs = self.ins_prs + smooth * self.init_ins_prs
        self.normalize()
    


class BigramModel:
    def __init__(self, words):
        bigram_counts = np.zeros((vocab.size, vocab.size))
        for word_ in words:
            word = '(' + word_ + ')'
            for i in range(1, len(word)):
                prev = word[i-1]
                bigram_counts[vocab.vocab2id[prev], vocab.vocab2id[word[i]]] += 1
        bigram_counts += 1e-5
        self.lm_probs = bigram_counts / bigram_counts.sum(axis=-1, keepdims=True)
        self.lm_probs = np.log(self.lm_probs)

    def string_pr(self, strings):
        return np.exp(self.string_logp(strings))

    def string_logp(self, strings):
        strings = [s + ')' for s in strings]
        prs = np.zeros(len(strings))
        for i, string in enumerate(strings):
            prev = vocab.vocab2id['(']
            for j in range(len(string)):
                cur = vocab.vocab2id[string[j]]
                prs[i] += self.lm_probs[prev, cur]
                prev = cur
        return prs

MODELS = {'Dirichlet': DirichletModel, 'Markedness': MarkednessModel}

