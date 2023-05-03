import numpy as np
from data import fr, pt, ro, es, it, la, small_dataset
from sampler import LeafNode, RootNode, mstep_tree, sample_tree, evaluate
from data import NATURAL_CLASSES, vocab
from models import DirichletModel, BigramModel, BaseModel
import wandb 
import argparse
from dp import compute_posteriors
import time 
import json 
import os 

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

def bag_with_replacement(list1, list2, bagging_rate):
    assert len(list1) == len(list2)
    indices = list(range(len(list1)))
    np.random.shuffle(indices)
    bag_size = int(len(list1) * bagging_rate)
    bag_indices = indices[:bag_size]
    bag1 = [list1[i] for i in bag_indices]
    bag2 = [list2[i] for i in bag_indices]
    return bag1, bag2

class SourceEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim 
        self.embed = nn.Embedding(vocab.size, embed_dim)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)

    def forward(self, encoder_input):
        embeddings = self.embed(encoder_input)
        encoder_output, hidden = self.encoder(embeddings)
        return (encoder_output[:, :, :self.embed_dim] + encoder_output[:, :, self.embed_dim:]) / 2
        # return encoder_output 

class SourceWindowEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, window_size):
        super().__init__()
        self.embed_dim = embed_dim 
        self.embed = nn.Embedding(vocab.size, embed_dim)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.window_size = window_size  # actually the radius

    def forward(self, encoder_input):
        embeddings = self.embed(encoder_input)
        outputs = []
        for index in range(encoder_input.shape[0]):
            left = max(index - self.window_size, 0)
            right = min(index + self.window_size + 1, len(encoder_input))
            window = embeddings[left:right]
            encoder_output, hidden = self.encoder(window)
            outputs.append(
                encoder_output[index-left:index-left+1]
            )
        encoder_output = torch.cat(outputs, dim=0)
        return (encoder_output[:, :, :self.embed_dim] + encoder_output[:, :, self.embed_dim:]) / 2
        

class TargetEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab.size, embed_dim)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers)
    
    def forward(self, encoder_input):
        embeddings = self.embed(encoder_input)
        encoder_output, hidden = self.encoder(embeddings)
        return encoder_output

class TargetWindowEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, window_size):
        super().__init__()
        self.embed = nn.Embedding(vocab.size, embed_dim)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.window_size = window_size

    def forward(self, encoder_input):
        embeddings = self.embed(encoder_input)
        outputs = []
        for index in range(encoder_input.shape[0]):
            left = max(index-self.window_size, 0)
            window = embeddings[left:index+1]
            encoder_output, hidden = self.encoder(window)
            outputs.append(
                encoder_output[index-left:index-left+1]
            )
        encoder_output = torch.cat(outputs, dim=0)
        return encoder_output

class InternalModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_head = nn.Linear(embed_dim, vocab.size, bias=False)
    
    def forward(self, src_encoding, tgt_encoding):
        encoding = src_encoding + tgt_encoding
        logits = self.cls_head(encoding)
        return logits


class LSTMModel(BaseModel):


    def __init__(self, embed_dim, window_size=None):

        self.aligner = DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2])
        self.baseline = DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2])
        
        if window_size is None:
            self.src_encoder = SourceEncoder(embed_dim, embed_dim, 1).cuda()
            self.tgt_encoder = TargetEncoder(embed_dim, embed_dim, 1).cuda()
        else:
            self.src_encoder = SourceWindowEncoder(embed_dim, embed_dim, 1, window_size).cuda()
            self.tgt_encoder = TargetWindowEncoder(embed_dim, embed_dim, 1, window_size).cuda()


        self.sub_model = InternalModel(embed_dim).cuda()
        self.ins_model = InternalModel(embed_dim).cuda()

        params = list(self.src_encoder.parameters()) + list(self.tgt_encoder.parameters()) \
            + list(self.sub_model.parameters()) + list(self.ins_model.parameters())
        
        self.optimizer = optim.Adam(params, lr=1e-2)


    def _cache_probs(self, sources, targets):
        batch_size = sources.shape[1]
        self.cache = TensorProbCache(sources, targets)
        sources = torch.from_numpy(sources).cuda()
        targets = torch.from_numpy(targets).cuda()
        
        src_encoding = self.src_encoder(sources) # L, N, D
        tgt_encoding = self.tgt_encoder(targets)


        for i in range(len(sources)):
            for j in range(len(targets)):
                src_char = src_encoding[i]
                tgt_char = tgt_encoding[j]

                sub_distrib = F.softmax(self.sub_model(src_char, tgt_char), dim=1).log()
                ins_distrib = F.softmax(self.ins_model(src_char, tgt_char), dim=1).log()

                if j+1 < len(targets):
                    # gather on gpu is faster than converting to numpy first then fancy indexing
                    gather_index = targets[j+1].unsqueeze(dim=1)
                    self.cache.sub[i, j] = torch.gather(sub_distrib, index=gather_index, dim=1)[:, 0]
                    self.cache.ins[i, j] = torch.gather(ins_distrib, index=gather_index, dim=1)[:, 0]
                self.cache.dlt[i, j] = sub_distrib[:, vocab.dlt]
                self.cache.end[i, j] = ins_distrib[:, vocab.dlt]

    def cache_probs(self, sources, targets, grad=False):
        if grad:
            self._cache_probs(sources, targets)
        else:
            with torch.no_grad():
                self._cache_probs(sources, targets)       

    def m_step(self, sources, targets, posterior_cache, num_epochs, bagging_rate):

        sources, targets = bag_with_replacement(sources, targets, bagging_rate)
        # override it 
        posterior_cache = compute_posteriors(self.aligner, sources, targets)
        
        raw_sources, raw_targets = sources, targets
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy()    
        batch_size = sources.shape[1]    
        
        sub_post = torch.from_numpy(np.exp(posterior_cache.sub)).cuda()
        dlt_post = torch.from_numpy(np.exp(posterior_cache.dlt)).cuda()
        ins_post = torch.from_numpy(np.exp(posterior_cache.ins)).cuda()
        end_post = torch.from_numpy(np.exp(posterior_cache.end)).cuda()


        for epoch in range(num_epochs):

            self.cache_probs(sources, targets, grad=True)
            sub_likelihood = (self.cache.sub * sub_post).sum() + (self.cache.dlt * dlt_post).sum()
            sub_likelihood /= batch_size
            ins_likelihood = (self.cache.ins * ins_post).sum() + (self.cache.end * end_post).sum()
            ins_likelihood /= batch_size

            self.optimizer.zero_grad()
            (-sub_likelihood - ins_likelihood).backward()
            self.optimizer.step() 
            
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


model_config = {'reverse_models':False, 'proposal_heuristic':True}

def evaluate_model(initialize_model, samples, labels):
    M = initialize_model
    italian = LeafNode(it, M(), 'IT', model_config)
    spanish = LeafNode(es, M(), 'ES', model_config)
    portuguese = LeafNode(pt, M(), 'PT', model_config)
    french = LeafNode(fr, M(), 'FR', model_config)
    romanian = LeafNode(ro, M(), 'RO', model_config)

    lm = BigramModel(la)
    root = RootNode([spanish, italian, portuguese, french], lm, 'LA', model_config)

    root.words = samples 
        
    for step, num_epochs in enumerate(MSTEP_SCHEDULE):
        mstep_tree(root, 
            num_epochs=num_epochs,
            bagging_rate=M_STEP_BAGGING_RATE) 
        
        train_log_pr = root.compute_likelihood(samples).mean()
        print('data likelihood', train_log_pr)
        gt_log_pr = root.compute_likelihood(labels).mean()
        print('gold recon likelihood', gt_log_pr)

        use_mh = not (GREEDY_SAMPLE or step == len(MSTEP_SCHEDULE)-1)
        sample_tree(root, mh=use_mh)
        recon_dist = evaluate(root.words, la)
        print('current recons: ', recon_dist)
        wandb.log({'data_likelihood':train_log_pr, 'gold_likelihood':gt_log_pr, 'recon_dist':recon_dist})

        with open(LOG_DIR + 'iteration%d.txt' % step, 'w') as f:
            f.writelines([s+'\n' for s in root.words])

    # sample_tree(root, mh=False)
    # print('final dist:', evaluate(root.words, la))

    # for rnd in range(rounds):
    #     root.words = samples 
    #     mstep_tree(root)        
    #     train_log_pr = root.compute_likelihood(samples).mean()
    #     print('data likelihood', train_log_pr)
    #     gt_log_pr = root.compute_likelihood(labels).mean()
    #     print('gold recon likelihood', gt_log_pr)
    #     # wandb.log({'data_likelihood':train_log_pr, 'gold_likelihood':gt_log_pr})
    
    #     sample_tree(root, mh=False)
    #     print('Final recons ', evaluate(root.words, la))

    # with open('experiments/final_recons_lstm.txt', 'w') as f:
    #     f.writelines([s+'\n' for s in root.words])

# wandb.init(project = 'recon_tuning', entity='andre-he')



# evaluate_model(
#     lambda : DirichletModel(INITIAL_PRS, NATURAL_CLASSES[2]),
#     samples, la, 1
# )

MSTEP_SCHEDULES = {
    '1': [8, 5, 5, 5, 5, 3, 3, 2, 2, 1],
    '2': [8, 5, 5, 5, 5, 3, 3, 3, 3, 3],
}


def main(config):
    if CONTEXT_WINDOW is None:
        initialize = lambda : LSTMModel(LSTM_DIM)
    else:
        initialize = lambda : LSTMModel(LSTM_DIM, int(CONTEXT_WINDOW))
    evaluate_model(
        initialize,
        samples, la
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lstm_dim', default=100)
    parser.add_argument('-b', '--m_step_bagging', default=1.0)
    parser.add_argument('-j', '--test_interval', default=5)   
    parser.add_argument('-i', '--initial_test_interval', default=8)
    parser.add_argument('-r', '--test_rounds', default=5)
    parser.add_argument('-t', '--debug', action='store_true', default=False)
    # parser.add_argument('-o', '--resample', action='store_true', default=False)
    parser.add_argument('-g', '--greedy_no_mh', action='store_true', default=False)
    parser.add_argument('-m', '--mstep_schedule', default=None)
    parser.add_argument('-n', '--name', default='-')
    parser.add_argument('-c', '--context_window', default=None)

    args = parser.parse_args()
    config = vars(args)
    print(config)

    LSTM_DIM = int(args.lstm_dim)
    M_STEP_BAGGING_RATE = float(args.m_step_bagging)
    TEST_ROUNDS = int(args.test_rounds)
    TEST_INTERVAL = int(args.test_interval)
    INIT_TEST_INTERVAL = int(args.initial_test_interval)
    # RESAMPLE = bool(args.resample)
    GREEDY_SAMPLE = bool(args.greedy_no_mh)
    RUN_NAME = args.name 
    CONTEXT_WINDOW = args.context_window

    if args.mstep_schedule is None:
        MSTEP_SCHEDULE = [INIT_TEST_INTERVAL] + (TEST_ROUNDS-1) * [TEST_INTERVAL]
    else:
        MSTEP_SCHEDULE = MSTEP_SCHEDULES[str(args.mstep_schedule)]

    LOG_DIR = 'log/' + RUN_NAME + '_' + str(int(time.perf_counter())) + '/'

    os.mkdir(LOG_DIR)
    with open(LOG_DIR + 'config.json', 'w') as f:
        json.dump(config, f)

    wandb.init(project = 'lstm_tuning', entity='andre-he', config=config)
    main(config)

