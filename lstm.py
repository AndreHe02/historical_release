import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
from data import cogs, vocab 
from torch.optim import Adam
from tqdm import tqdm
from evaluate import batch_edit_dist
from align import align

use_wandb = False 

# LSTM encoder-decoder, with hard monotonic attention
# Reimplementation of model from 
# Morphological Inflection Generation with Hard Monotonic Attention
class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.enc_hidden_dim = config['enc_hidden_dim']
        self.enc_num_layers = config['enc_num_layers']
        self.embed_dim = config['embed_dim']

        self.embed = nn.Embedding(vocab.size, self.embed_dim)
        self.encoder = nn.LSTM(input_size=self.embed_dim, hidden_size=self.enc_hidden_dim, 
            num_layers=self.enc_num_layers, bidirectional=True) # can add dropout here

    def forward(self, encoder_input):
        embedding = self.embed(encoder_input)
        encoder_mask = encoder_input == vocab.pad
        encoder_output, hidden = self.encoder(embedding)  
        return encoder_output, encoder_mask # we don't use the hidden state

class EditDecoder(nn.Module):

    def __init__(self, config, embed):    # pass embedding layer in
        super().__init__()
        self.enc_hidden_dim = config['enc_hidden_dim']
        self.enc_num_layers = config['enc_num_layers']
        self.dec_hidden_dim = config['dec_hidden_dim']
        self.dec_num_layers = config['dec_num_layers']
        self.embed_dim = config['embed_dim']

        self.embed = embed

        self.dec_input_size = 2*self.enc_hidden_dim + self.embed_dim
        self.decoder = nn.LSTM(input_size=self.dec_input_size, hidden_size=self.dec_hidden_dim, 
            num_layers=self.dec_num_layers)  # dropout=0.5
        self.cls_head = nn.Linear(self.dec_hidden_dim, vocab.size)
        

    def forward(self, decoder_input, aligned_encodings, initial_hidden=None):
        embedding = self.embed(decoder_input) # S, N, E
        decoder_input = torch.cat((embedding, aligned_encodings), dim=2)    # (S, N, 2H+E)
        if initial_hidden is None:
            decoder_output, hidden = self.decoder(decoder_input)    # (S, N, H)
        else:
            decoder_output, hidden = self.decoder(decoder_input, initial_hidden)
        logits = self.cls_head(decoder_output)  # (S, N, V)
        return logits, hidden

class HardAttentionSeq2Seq(nn.Module):

    # self.embed_dim = 100, hidden_size = 100, num_layers=2, bidirectional
    def __init__(self, config, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = LSTMEncoder(config)
            self.decoder = EditDecoder(config, self.encoder.embed)
        else: 
            self.encoder = encoder
            self.decoder = EditDecoder(config, encoder.embed)


    def encode(self, encoder_input):
        embedding = self.embed(encoder_input)
        encoder_mask = encoder_input == vocab.pad
        encoder_output, hidden = self.encoder(embedding)
        # we don't use the hidden state in this model; decode directly from contextualized embeddings
        # encoder_output is L, N, 2H
        return encoder_output, encoder_mask

    def decode(self, decoder_input, aligned_encodings, initial_hidden=None):   #remember to deal with encoder mask 
        # decoder_input (S, N); hidden (1, H; 1, H); 
        # encoder_output (S, N, 2H)  
        embedding = self.embed(decoder_input) # S, N, E
        decoder_input = torch.cat((embedding, aligned_encodings), dim=2)    # (S, N, 2H+E)
        if initial_hidden is None:
            decoder_output, hidden = self.decoder(decoder_input)    # (S, N, H)
        else:
            decoder_output, hidden = self.decoder(decoder_input, initial_hidden)
        logits = self.cls_head(decoder_output)  # (S, N, V)
        return logits, hidden

    def compute_loss(self, encoder_input, target):
        batch_size = encoder_input.shape[1]
        encoder_output, encoder_mask = self.encoder(encoder_input)
        # encoder_output, encoder_mask = self.encode(encoder_input)   # L, N, 2H
        aligned_encodings = torch.zeros((len(target), *encoder_output.shape[1:])).cuda()
        cur_index = torch.zeros(batch_size, dtype=torch.long).cuda()
        for i in range(len(target)):
            cur_index = cur_index + (target[i] == vocab.step)
            cur_index = torch.minimum(cur_index, 
                torch.full_like(cur_index, fill_value=len(target)-1))
            aligned_encodings[i] = encoder_output[cur_index, range(batch_size)]
        # logits, _ = self.decode(target[:-1], aligned_encodings[:-1])
        logits, _ = self.decoder(target[:-1], aligned_encodings[:-1])
        target_mask = target[1:] == vocab.pad
        logits[:, :, vocab.pad] += target_mask * 1e9   # force output to be pad so it incurs no loss
        loss = F.cross_entropy(logits.permute(1, 2, 0), target[1:].permute(1, 0))
        return loss 
        
    def output_prob(self, sources, targets):
        with torch.no_grad():
            encoder_input = vocab.make_tensor(sources).cuda()
            targets = vocab.make_tensor(targets).cuda()
            encoder_output, encoder_mask = self.encoder(encoder_input)
            # encoder_output, encoder_mask = self.encode(encoder_input)   # L, N, 2H
            aligned_encodings = torch.zeros((len(target), *encoder_output.shape[1:])).cuda()
            cur_index = torch.zeros(batch_size, dtype=torch.long).cuda()
            for i in range(len(target)):
                cur_index = cur_index + (target[i] == vocab.step)
                cur_index = torch.minimum(cur_index, 
                    torch.full_like(cur_index, fill_value=len(target)-1))
                aligned_encodings[i] = encoder_output[cur_index, range(batch_size)]
            # logits, _ = self.decode(target[:-1], aligned_encodings[:-1])
            logits, _ = self.decoder(target[:-1], aligned_encodings[:-1])
            target_mask = target[1:] == vocab.pad
            logits[:, :, vocab.pad] += target_mask * 1e9   # force output to be pad so it incurs no loss
            logps = F.softmax(logits, dim=-1)
            
            
            loss = F.cross_entropy(logits.permute(1, 2, 0), target[1:].permute(1, 0))
            return loss 



    def predict_greedy(self, inputs, max_length=30):
        with torch.no_grad():
            batch_size = len(inputs)
            encoder_input = vocab.make_tensor(inputs).cuda()
            # encoder_output, encoder_mask = self.encode(encoder_input)
            encoder_output, encoder_mask = self.encoder(encoder_input)
            next_word = torch.full((1, batch_size), fill_value=vocab.sow, dtype=torch.long).cuda()
            terminated = torch.zeros(1, batch_size).cuda()
            predictions = [next_word.detach().cpu().numpy()]
            hidden = (torch.zeros((self.decoder.decoder.num_layers, batch_size, self.decoder.dec_hidden_dim)).cuda(), 
                    torch.zeros((self.decoder.decoder.num_layers, batch_size, self.decoder.dec_hidden_dim)).cuda())
            cur_index = torch.zeros(batch_size, dtype=torch.long).cuda()

            for step in range(max_length):
                aligned_encodings = encoder_output[cur_index, range(batch_size), :]
                aligned_encodings = aligned_encodings.unsqueeze(dim=0)
                # next_word_logits, hidden = self.decode(next_word, aligned_encodings, hidden)
                next_word_logits, hidden = self.decoder(next_word, aligned_encodings, hidden)
                
                next_word_logits[:, :, vocab.pad] += terminated * 1e9  # force EOS
                next_word = torch.argmax(next_word_logits, dim=-1)
                take_step = next_word == vocab.step
                cur_index += take_step[0]
                # shouldnt terminate before taking L steps; shouldn't take more than L steps. 
                # these rules are not enforced as of now
                cur_index = torch.minimum(cur_index, torch.full_like(cur_index, fill_value=len(encoder_output)-1))
                
                terminated = terminated + (next_word == vocab.eow)
                predictions.append(next_word.detach().cpu().numpy())

            predictions = np.concatenate(predictions, axis=0).T
            predictions = [vocab.ids2string(w) for w in predictions]
        return predictions

    # beam search 

def train(model, dataset, batch_size, optimizer=None, test=False):
    if test:
        model.eval()
    else:
        model.train()
        np.random.shuffle(dataset)
    inputs = [d[0] for d in dataset]
    targets = [d[1] for d in dataset]

    acc_loss = 0
    for i in range(0, len(inputs), batch_size):
        source = inputs[i:i + batch_size]
        target = targets[i:i + batch_size]

        source = vocab.make_tensor(source).cuda()
        target = vocab.make_tensor(target).cuda()
        loss = model.compute_loss(source, target)
        if not test:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc_loss += loss.item()
    return acc_loss / len(inputs)

def test(model, dataset, batch_size):
    with torch.no_grad():
        return train(model, dataset, batch_size, test=True)

def evaluate(model, dataset, batch_size):
    model.eval()
    inputs = [d[0] for d in dataset]
    targets = [d[1] for d in dataset]
    accum_dist = 0
    for i in range(0, len(inputs), batch_size):
        source = inputs[i:i + batch_size]
        target = targets[i:i + batch_size]
        raw_target = [''.join(x for x in t if x.isalnum()) for t in target]
        pred = model.predict_greedy(source)
        raw_pred = [''.join(x for x in t if x.isalnum()) for t in pred]
       
        accum_dist += batch_edit_dist(raw_target, raw_pred).sum()
    return accum_dist / len(inputs)

def make_dataset(inputs, outputs):
    inputs = ['('+w for w in inputs]
    outputs = ['('+w for w in outputs]
    aligned = align(inputs, outputs)
    return [w+')' for w in inputs], [w+')' for w in aligned]

def train_edit_model(model, run_config, train_set, test_set):
    lr = run_config['lr']
    batch_size = run_config['batch_size']
    epochs = run_config['epochs']
    path = 'checkpoints/' + run_config['name']

    optimizer = Adam(model.parameters(), lr)

    best_loss = 100.0
    for e in range(epochs): #tqdm(range(epochs)):
        train_loss = train(model, train_set, batch_size, optimizer=optimizer)
        test_loss = test(model, test_set, batch_size)

        if e % 10 == 0:
            print('Train and test loss', train_loss, test_loss)
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), path)
        dist = evaluate(model, test_set, batch_size)

        if e % 10 == 0:
            print('Greedy prediction dist', dist)

        if use_wandb:
            wandb.log({'train_loss':train_loss, 'test_loss':test_loss, 'test_dist':dist})

    model.load_state_dict(torch.load(path))

def main(run_config, model_config):
    source_lang = run_config['source']
    target_lang = run_config['target']

    inputs = [cog[source_lang] for cog in cogs]
    outputs = [cog[target_lang] for cog in cogs]
    inputs, outputs = make_dataset(inputs, outputs)
    
    
    data_indices = list(range(len(inputs)))
    np.random.seed(0)
    np.random.shuffle(data_indices)
    test_indices = data_indices[:len(inputs)//10]
    train_indices = data_indices[len(inputs)//10:]

    test_inputs = [inputs[i] for i in test_indices]
    test_outputs = [outputs[i] for i in test_indices]
    train_inputs = [inputs[i] for i in train_indices]
    train_outputs = [outputs[i] for i in train_indices]
    
    test_set = list(zip(test_inputs, test_outputs))
    train_set = list(zip(train_inputs, train_outputs))

    model = HardAttentionSeq2Seq(model_config).cuda()
    train_edit_model(model, run_config, train_set, test_set)

# gpu "python lstm.py -s 3 -t 2 -n es2it -h1 100 -h2 100 -l1 2 -l2 2 -e 100"
if __name__ == "__main__":
    use_wandb = True
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True)
    parser.add_argument('-t', '--target', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-l', '--learning_rate', default=0.01)
    parser.add_argument('-h1', '--enc_hidden_dim', required=True)
    parser.add_argument('-h2', '--dec_hidden_dim', required=True)
    parser.add_argument('-l1', '--enc_num_layers', required=True)
    parser.add_argument('-l2', '--dec_num_layers', required=True)
    parser.add_argument('-e', '--embed_dim', required=True)
    args = parser.parse_args()


    run_config = {  'source': int(args.source),
                    'target': int(args.target),
                    'lr': float(args.learning_rate),
                    'batch_size': 512,
                    'epochs': 400,
                    'name': args.name}

    model_config = {'enc_hidden_dim': int(args.enc_hidden_dim),
                    'enc_num_layers': int(args.enc_num_layers),
                    'dec_hidden_dim': int(args.dec_hidden_dim),
                    'dec_num_layers': int(args.dec_num_layers),
                    'embed_dim': int(args.embed_dim)}
    
    if use_wandb:  
        config = {**run_config, **model_config}
        wandb.init(project='historical3', entity='andre-he', config=config)

    main(run_config, model_config)

