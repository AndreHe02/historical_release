import numpy as np
from data import fr, pt, ro, es, it, la
from sampler import LeafNode, RootNode, mstep_tree, sample_tree, evaluate
from data import NATURAL_CLASSES, vocab
from models import DirichletModel, BigramModel, BaseModel
import wandb 

def read_outputs(name, version=None):
    version_name = 'iteration%d.txt'%version if version is not None else 'final_recons.txt'
    with open('log/' + name +'/' +version_name, 'r') as f:
        lines = f.readlines()
        return [s.strip() for s in lines]

runs = ['no_cls_8845807', 'vow_con_cls_8848414', 'ipa_ft_cls_8851033', 'id_cls_8853651']
sample_sets = [read_outputs(run, 3) for run in runs]
samples = []        # mix up samples from different runs
for i in range(len(sample_sets[0])):
    choice = np.random.choice([s[i] for s in sample_sets])
    samples.append(choice)

## To quickly evaluate models, we want to fit them to a typical batch of samples
## and check the likelihood that they assign to true Latin words.


INITIAL_PRS = (0.7, 0.3/vocab.size, 1-0.3/vocab.size)
# natural_classes = NATURAL_CLASSES[0]
# train_likelihood = evaluate_model(
#     lambda : DirichletModel(INITIAL_PRS, natural_classes),
#     samples, samples)
# label_likelihood = evaluate_model(
#     lambda : DirichletModel(INITIAL_PRS, natural_classes),
#     samples, la )
# print(train_likelihood, label_likelihood)

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from models import ProbCache, TensorProbCache

# maybe ensemble this with Dirichlet model to start training
# have embedding model predict zero-sum vector over vocab. add to base model's distribution

class EmbeddingModel(BaseModel):
    
    class InternalModel(nn.Module):

        # def __init__(self, embed_dim):
        #     super().__init__()
        #     self.logits = nn.Parameter(torch.zeros((vocab.size, vocab.size, vocab.size+1)))
        
        # def forward(self, src_char, prev_tgt_char):
        #     logits = self.logits[src_char, prev_tgt_char, :]
        #     return logits 

        def __init__(self, embed_dim):
            super().__init__()
            self.src_embed = nn.Embedding(vocab.size, embed_dim)
            self.tgt_embed = nn.Embedding(vocab.size, embed_dim)
            self.cls_head = nn.Linear(embed_dim, vocab.size, bias=False)

        def forward(self, src_char, prev_tgt_char):
            src_embeddings = self.src_embed(src_char)
            tgt_embeddings = self.tgt_embed(prev_tgt_char)
            embeddings = src_embeddings + tgt_embeddings
            logits = self.cls_head(embeddings)
            return logits 

        # def compute_loss(self, src_char, prev_tgt_char, tgt_char):
            
        #     if isinstance(tgt_char, int):
        #         tgt_char = torch.full_like(src_char, fill_value=tgt_char).cuda()
        #     logits = self.forward(src_char, prev_tgt_char)

        #     return F.cross_entropy(logits, tgt_char)

    def __init__(self, embed_dim):

        self.aligner = DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2])
        self.baseline = DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2])
        
        self.sub_model = self.InternalModel(embed_dim).cuda()
        self.ins_model = self.InternalModel(embed_dim).cuda()

    def _cache_probs(self, sources, targets):
        batch_size = sources.shape[1]
        self.cache = TensorProbCache(sources, targets)
        sources = torch.from_numpy(sources).cuda()
        targets = torch.from_numpy(targets).cuda()

        for i in range(len(sources)):
            for j in range(len(targets)):
                sub_distrib = F.softmax(self.sub_model(sources[i], targets[j]), dim=1).log()
                ins_distrib = F.softmax(self.ins_model(sources[i], targets[j]), dim=1).log()

                if j+1 < len(targets):
                    # gather on gpu is faster than converting to numpy first then fancy indexing
                    gather_index = targets[j+1].unsqueeze(dim=1)
                    self.cache.sub[i, j] = torch.gather(sub_distrib, index=gather_index, dim=1)[:, 0]
                    self.cache.ins[i, j] = torch.gather(ins_distrib, index=gather_index, dim=1)[:, 0]
                self.cache.dlt[i, j] = sub_distrib[:, vocab.dlt]
                self.cache.end[i, j] = ins_distrib[:, vocab.dlt]

        
        
    # def _cache_probs(self, sources, targets):
    #     batch_size = sources.shape[1]
    #     self.cache = TensorProbCache(sources, targets)
    #     sources = torch.from_numpy(sources).cuda()
    #     targets = torch.from_numpy(targets).cuda()

    #     for i in range(len(sources)):
    #         for j in range(len(targets)):
    #             sub_distrib = F.softmax(self.sub_model(sources[i], targets[j]), dim=1).log()    # might need to cache full distribution for ensemble
    #             ins_distrib = F.softmax(self.ins_model(sources[i], targets[j]), dim=1).log()

    #             if j+1 < len(targets):
    #                 # gather on gpu is faster than converting to numpy first then fancy indexing
    #                 gather_index = targets[j+1].unsqueeze(dim=1)
    #                 self.cache.sub[i, j] = torch.gather(sub_distrib, index=gather_index, dim=1)[:, 0]
    #                 self.cache.ins[i, j] = torch.gather(ins_distrib, index=gather_index, dim=1)[:, 0]
    #             self.cache.dlt[i, j] = sub_distrib[:, vocab.dlt]
    #             self.cache.end[i, j] = ins_distrib[:, vocab.dlt]

    #     self.cache.sub = self.cache.sub * 

    def cache_probs(self, sources, targets, grad=False):

        if grad:
            self._cache_probs(sources, targets)
        else:
            with torch.no_grad():
                self._cache_probs(sources, targets)
        

    def m_step(self, sources, targets, posterior_cache):
        

        raw_sources, raw_targets = sources, targets
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy()    
        batch_size = sources.shape[1]    
        
        sub_post = torch.from_numpy(np.exp(posterior_cache.sub)).cuda()
        dlt_post = torch.from_numpy(np.exp(posterior_cache.dlt)).cuda()
        ins_post = torch.from_numpy(np.exp(posterior_cache.ins)).cuda()
        end_post = torch.from_numpy(np.exp(posterior_cache.end)).cuda()


        # sub_optimizer = optim.SGD(self.sub_model.parameters(), lr=1e-2)
        # ins_optimizer = optim.SGD(self.ins_model.parameters(), lr=1e-2)

        sub_optimizer = optim.Adam(self.sub_model.parameters(), lr=1e-2)
        ins_optimizer = optim.Adam(self.ins_model.parameters(), lr=1e-2)

        # simple solution: train on entire word set, validation by
        # looking at likelihood of latin 

        # TODO don't use cross entropy loss. optimize likelihood directly so 
        # train and inference are consistent 
        for epoch in range(5):

            # sub_model_loss = 0.0
            # ins_model_loss = 0.0

            self.cache_probs(sources, targets, grad=True)
            sub_likelihood = (self.cache.sub * sub_post).sum() + (self.cache.dlt * dlt_post).sum()
            sub_likelihood /= batch_size
            ins_likelihood = (self.cache.ins * ins_post).sum() + (self.cache.end * end_post).sum()
            ins_likelihood /= batch_size
            
            sub_optimizer.zero_grad()
            (-sub_likelihood).backward()
            sub_optimizer.step()

            ins_optimizer.zero_grad()
            (-ins_likelihood).backward()
            ins_optimizer.step() 

            # print((sub_likelihood + ins_likelihood).item())
          
        self.aligner = self 


    def sub(self, i, j):
        return self.cache.sub[i, j].cpu().numpy()
    
    def ins(self, i, j):
        return self.cache.ins[i, j].cpu().numpy()

    def dlt(self, i, j):
        return self.cache.dlt[i, j].cpu().numpy()

    def end(self, i, j):
        return self.cache.end[i, j].cpu().numpy()


config = {'reverse_models':False, 'proposal_heuristic':True}

def evaluate_model(initialize_model, samples, labels, rounds):
    M = initialize_model
    italian = LeafNode(it, M(), 'IT', config)
    spanish = LeafNode(es, M(), 'ES', config)
    portuguese = LeafNode(pt, M(), 'PT', config)
    french = LeafNode(fr, M(), 'FR', config)
    romanian = LeafNode(ro, M(), 'RO', config)

    lm = BigramModel(la)
    root = RootNode([spanish, italian, portuguese, french], lm, 'LA', config)

    root.words = samples 

    for rnd in range(rounds):
        mstep_tree(root)        
        train_log_pr = root.compute_likelihood(samples).mean()
        print('data likelihood', train_log_pr)
        gt_log_pr = root.compute_likelihood(labels).mean()
        print('gold recon likelihood', gt_log_pr)
        # wandb.log({'data_likelihood':train_log_pr, 'gold_likelihood':gt_log_pr})
    
    sample_tree(root, mh=False)
    print('Final recons ', evaluate(root.words, la))
    with open('experiments/final_recons.txt', 'w') as f:
        f.writelines([s+'\n' for s in root.words])

# wandb.init(project = 'recon_tuning', entity='andre-he')


evaluate_model(
    lambda : EmbeddingModel(100),
    samples, la, 1
)

# evaluate_model(
#     lambda : DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2]),
#     samples, la, 1
# )

