from models import DirichletModel, BigramModel, MODELS
from dp import compute_mutation_prob, compute_posteriors
from proposal import heuristic_proposals, one_edit_proposals
from data import cogs, vocab
from data import DEGEN_CLASSES, VOW_CON_CLASSES, IPA_FT_CLASSES, IDENTITY_CLASSES
from evaluate import batch_edit_dist, edit_distance
from tqdm import tqdm 
import numpy as np 
import argparse
import time 
import json
import os 

# from embed_model import EmbeddingModel

# from multiprocessing.pool import ThreadPool as Pool 
# from tqdm.contrib.concurrent import process_map
# from tqdm import tqdm 

from data import fr, pt, ro, es, it, la, small_dataset

class LanguageNode:

    def __init__(self, children, model, name, config):
        self.config = config
        self.children = children
        for child in self.children:
            child.top = self
        self.top = None 
        self.model = model 
        self.is_leaf = False
        self.is_root = False 
        self.name = name 
        self.initialize_samples()

    def initialize_samples(self):
        words = []
        for i in range(len(self.children[0].words)):
            leaves = [child.words[i] for child in self.children]
            words.append(np.random.choice(leaves))
        self.words = words

    def sample_recon(self, mh, i):
        cur = self.words[i]
        sample_likelihood = None

        num_rounds = 1 if bool(self.config['proposal_heuristic']) else 5

        for _ in range(num_rounds):
            leaves = [child.words[i] for child in self.children]
            if bool(self.config['proposal_heuristic']):
                proposals = heuristic_proposals(cur, leaves)
                if len(proposals) > 500:
                    proposals = np.random.choice(proposals, size=500, replace=False)
            else:
                proposals = one_edit_proposals(cur, leaves)
            
            proposal_prs = np.zeros(len(proposals))

            for child in self.children:
                leaf = child.words[i]
                if not bool(self.config['reverse_models']):
                    branch_pr = compute_mutation_prob(child.model, proposals, [leaf]*len(proposals))
                else:
                    branch_pr = compute_mutation_prob(child.model, [leaf]*len(proposals), proposals)                
                proposal_prs += branch_pr
            
            if self.is_root:
                lm_pr = self.lm.string_logp(proposals)
                proposal_prs += lm_pr 
            else:
                if not bool(self.config['reverse_models']):
                    top_pr = compute_mutation_prob(self.model, [self.top.words[i]]*len(proposals), proposals)
                else:
                    top_pr = compute_mutation_prob(self.model, proposals, [self.top.words[i]]*len(proposals))
                proposal_prs += top_pr 

            if not mh:
                argmax = np.argmax(proposal_prs)
                sample_likelihood = proposal_prs[argmax]
                if proposals[argmax] == cur:
                    break 
                cur = proposals[argmax]
            else:
                prs = np.exp(proposal_prs-proposal_prs.max()) / np.exp(proposal_prs-proposal_prs.max()).sum()
                idx = np.random.choice(range(len(proposals)), p=prs)
                cur = proposals[idx]
                sample_likelihood = proposal_prs[idx]

        return cur, sample_likelihood
        

    def sample_reconstructions(self, mh):
        print('Sampling reconstructions for %s' % self.name)
        recons = []
        likelihood = 0.0

        for i in tqdm(range(len(self.words)), leave=False):
            sample, l = self.sample_recon(mh, i)
            recons.append(sample)
            likelihood += l 

        # with Pool(32) as pool:
        #     for sample, l in tqdm(pool.imap_unordered(
        #         lambda i: self.sample_recon(mh, i), 
        #         range(len(self.words)), chunksize=10)):
        #         recons.append(sample)
        #         likelihood += l 

        # results = process_map(
        #     lambda i: self.sample_recon(mh, i), 
        #     range(len(self.words)), 
        #     max_workers=32, 
        #     chunksize=100)
    
        self.words = recons
        return likelihood

    def compute_likelihood(self, samples):
        log_prs = np.zeros(len(samples))
        for child in self.children:
            if not bool(self.config['reverse_models']):
                branch_pr = compute_mutation_prob(child.model, samples, child.words)
            else:
                branch_pr = compute_mutation_prob(child.model, child.words, samples)
            log_prs += branch_pr 
            assert all(log_prs <= 0)
        if self.is_root:    # will also include lm for consistency
            lm_pr = self.lm.string_logp(samples)
            log_prs += lm_pr
            # assert all(log_prs <= 0)

        return log_prs 


    def update_model(self, **kwargs):
        if self.is_root:
            self.lm = BigramModel(self.words)
        else:
            # print('M step for %s' % self.name)
            if not bool(self.config['reverse_models']):
                posteriors = compute_posteriors(self.model.aligner, self.top.words, self.words)
                self.model.m_step(self.top.words, self.words, posteriors, **kwargs)
            else:
                posteriors = compute_posteriors(self.model.aligner, self.words, self.top.words)
                self.model.m_step(self.words, self.top.words, posteriors, **kwargs)


class LeafNode(LanguageNode):
    def __init__(self, words, model, name, config):
        self.config = config
        self.words = words
        self.model = model 
        self.is_leaf = True
        self.is_root = False 
        self.name = name 
        self.top = None 

class RootNode(LanguageNode):
    def __init__(self, children, lm, name, config):
        self.config = config 
        self.children = children
        for child in self.children:
            child.top = self
        self.lm = lm 
        self.is_root = True
        self.is_leaf = False 
        self.name = name
        self.initialize_samples()

def sample_tree(node, mh):
    if node.is_leaf:
        return 0
    likelihood = 0
    for child in node.children:
        lc = sample_tree(child, mh)
        likelihood += lc
    likelihood += node.sample_reconstructions(mh=mh)
    return likelihood

def mstep_tree(node, **kwargs):
    if not node.is_leaf:
        for child in node.children:
            mstep_tree(child, **kwargs)
    if not node.is_root:
        node.update_model(**kwargs)

def evaluate(recons, la):
    return batch_edit_dist(recons, la).mean()


def run_EM(root):
    hist = []
    for itr in range(EM_ROUNDS):
        # for i in range(5):
            # print('Burn in %d' % i)
        likelihood = sample_tree(root, mh=True)
        print(likelihood)
        dist = evaluate(root.words, la)
        print(dist)
        hist.append(dist)

        if LOG_OUTPUT:
            with open(LOG_DIR + 'iteration%d.txt' % itr, 'w') as f:
                f.writelines([s+'\n' for s in root.words])
        mstep_tree(root)
    print(hist)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reverse_models', action='store_true', default=False)
    parser.add_argument('-e', '--em_rounds', default=5)
    parser.add_argument('-p', '--proposal_heuristic', action='store_true', default=False)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-c', '--conditioning_type', default=0)
    parser.add_argument('-l', '--lm_percent', default=1.0)
    parser.add_argument('-f', '--flat_tree', action='store_true', default=False)
    parser.add_argument('-s', '--small_dataset', default=None)
    parser.add_argument('-k', '--use_cluster_model', action='store_true', default=False)
    parser.add_argument('-d', '--distance_threshold', default=10.0)
    parser.add_argument('-o', '--log_output', action='store_true', default=False)
    parser.add_argument('-m', '--model', required=True)


    args = parser.parse_args()
    config = vars(args)
    print(config)

    # bool(config['proposal_heurisitc']) = bool(args.proposal_heuristic)
    # REVERSE_MODELS = bool(args.reverse_models)
    RUN_NAME = args.name 
    LOG_DIR = 'log/' + RUN_NAME + '_' + str(int(time.perf_counter())) + '/'
    EM_ROUNDS = int(args.em_rounds)
    LM_PERCENT = float(args.lm_percent)
    CONDITIONING_TYPE = int(args.conditioning_type)
    USE_FLAT_TREE = bool(args.flat_tree)
    USE_CLUSTER_MODEL = bool(args.use_cluster_model)
    CLUSTERING_DIST_THRESH = float(args.distance_threshold)
    LOG_OUTPUT = bool(args.log_output)
    MODEL_NAME = args.model

    natural_classes = [DEGEN_CLASSES, VOW_CON_CLASSES, IPA_FT_CLASSES, IDENTITY_CLASSES][CONDITIONING_TYPE]
    all_la = la
    if args.small_dataset is not None:
        langs, cogs = small_dataset(int(args.small_dataset))
        fr, pt, ro, es, it, la = langs

    if LOG_OUTPUT:
        print('Writing log into', LOG_DIR)
        os.mkdir(LOG_DIR)
        with open(LOG_DIR + 'config.json', 'w') as f:
            json.dump(config, f)


    lm_data = all_la.copy()
    np.random.shuffle(lm_data)
    lm_data = lm_data[:int(len(lm_data) * LM_PERCENT)]
    lm = BigramModel(lm_data)

    INITIAL_PRS = (0.7, 0.3/vocab.size, 1-0.3/vocab.size)
    
    ModelClass = MODELS[MODEL_NAME]
    if MODEL_NAME == 'Dirichlet':       # arguments might be different
        M = lambda : ModelClass(INITIAL_PRS, natural_classes)
    elif MODEL_NAME == 'Markedness':
        M = lambda : ModelClass(INITIAL_PRS, natural_classes)
    elif MODEL_NAME == 'Embedding':
        M = lambda: EmbeddingModel(100)
    else:
        raise Exception
    

    italian = LeafNode(it, M(), 'IT', config)
    spanish = LeafNode(es, M(), 'ES', config)
    portuguese = LeafNode(pt, M(), 'PT', config)
    french = LeafNode(fr, M(), 'FR', config)
    romanian = LeafNode(ro, M(), 'RO', config)

    if USE_FLAT_TREE:
        # root = RootNode([spanish, italian, portuguese, french, romanian], lm, 'LA', config)
        root = RootNode([spanish, italian, portuguese, french], lm, 'LA', config)

    else:
        ibero = LanguageNode([spanish, portuguese], M(), 'IBERO', config)
        western = LanguageNode([ibero, french], M(), 'WESTERN', config)
        italo = LanguageNode([western, italian], M(), 'ITALO', config)
        root = RootNode([italo, romanian], lm, 'LA', config)
        # root = RootNode([ibero, italian], lm, 'LA', config)

    run_EM(root)

    sample_tree(root, mh=False)
    if LOG_OUTPUT:
        print('Final recons ', evaluate(root.words, la))
        with open(LOG_DIR + 'final_recons.txt', 'w') as f:
            f.writelines([s+'\n' for s in root.words])

