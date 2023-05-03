import re 
import numpy as np 
import torch 

def remove_stress(word):
    word = word.replace('ˈ', '')
    word = word.replace('ˌ', '')
    return word

def remove_length_indicators(word):
    word = word.replace(':', '')
    word = word.replace('ː', '')
    word = word.replace('-', '')
    return word 

def remove_parentheses_labels(word):
    return re.sub("[\(\[].*?[\)\]]", "", word)

def remove_superscripts(word):
    word = word.replace('ʲ', '')
    word = word.replace('ʰ', '')
    word = word.replace('ɔ̃'[1], '')
    return word 

# add ()?

def preprocess_ipa(words):
    processed = []
    for word in words:
        w = remove_stress(word)
        w = remove_length_indicators(w)
        w = remove_parentheses_labels(w)
        w = remove_superscripts(w)
        processed.append(w)
    return processed

def read_words(filename):
    with open(filename, 'r') as f:
        words = f.readlines()
        words = [w.strip() for w in words]
    return words

fr = preprocess_ipa(read_words('data/French_ipa.txt'))
pt = preprocess_ipa(read_words('data/Portuguese_ipa.txt'))
ro = preprocess_ipa(read_words('data/Romanian_ipa.txt'))
es = preprocess_ipa(read_words('data/Spanish_ipa.txt'))
it = preprocess_ipa(read_words('data/Italian_ipa.txt'))
la = preprocess_ipa(read_words('data/Latin_ipa.txt'))
langs = (fr, pt, ro, es, it, la)
cogs = [[lang[i] for lang in langs] for i in range(len(fr))]

def small_dataset(per_n):
    subset_indices = list(range(0, len(la), per_n))
    fr_sm = [fr[i] for i in subset_indices]
    pt_sm = [pt[i] for i in subset_indices]
    ro_sm = [ro[i] for i in subset_indices]
    es_sm = [es[i] for i in subset_indices]
    it_sm = [it[i] for i in subset_indices]
    la_sm = [la[i] for i in subset_indices]
    langs_sm = (fr_sm, pt_sm, ro_sm, es_sm, it_sm, la_sm)
    cogs_sm = [[lang[i] for lang in langs_sm] for i in range(len(fr_sm))]
    return langs_sm, cogs_sm

vows = {}
cons = {}
with open("converted-vow.ipa.txt", "r") as f:
    lines = f.readlines()
for line in lines:
    phoneme, *fts = line.split()
    vows[phoneme] = ['vowel'] + fts
with open("converted-cons.ipa.txt", "r") as f:
    lines = f.readlines()
for line in lines:
    phoneme, *fts = line.split()
    cons[phoneme] = ['consonant'] + fts 


class Vocabulary:
    def __init__(self, words):
        vocab = set()
        for word in words:
            for x in word:
                vocab.add(x)
        vocab = sorted(list(vocab))
        vocab.append('-')
        # vocab.append('#')
        vocab.append('|')
        vocab.append('(')
        vocab.append(')')
        self.vocab = vocab
        self.vocab2id = {x: i for i, x in enumerate(vocab)}
        self.pad = self.vocab2id['-']
        # self.boundary_id = self.vocab2id['#']
        self.step = self.vocab2id['|']
        self.dlt = self.step
        self.sow = self.vocab2id['(']
        self.eow = self.vocab2id[')']
        self.size = len(vocab)

        self.vows = vows
        self.cons = cons 

    def string2ids(self, s):
        return [self.vocab2id[x] for x in s]
    
    def ids2string(self, ids):
        s = ''.join([self.vocab[i] for i in ids])
        s = s.replace('-', '')
        s = s.replace('#', '')
        return s

    def make_batch(self, words, align_right=False):
        lens = [len(w) for w in words]
        max_len = max(lens)
        batch = np.full((max_len, len(words)), fill_value=self.pad, dtype=np.int32)
        for i, w in enumerate(words):
            for j, x in enumerate(w):
                if align_right:
                    batch[max_len - len(w) + j, i] = self.vocab2id[x]
                else:
                    batch[j, i] = self.vocab2id[x]
        return batch, np.array(lens)

    def make_tensor(self, words, add_boundaries=False):
        if add_boundaries:
            words = ['(%s)'%w for w in words]
        np_batch, lens = self.make_batch(words)
        np_batch = np_batch.astype(np.long)
        return torch.from_numpy(np_batch) #.cuda()
    

vocab = Vocabulary(fr+pt+ro+es+it+la)

"""Assignments of phonemes to classes used for conditioning"""

DEGEN_CLASSES = np.zeros(vocab.size, dtype=np.int32)

VOW_CON_CLASSES = np.zeros(vocab.size, dtype=np.int32)
for v in vocab.vocab:
    i = vocab.vocab2id[v]
    if v in ['#', '|', '-', '(', ')']:   # shouldnt ever see these 
        VOW_CON_CLASSES[i] = 0
    elif v in vocab.vows:
        VOW_CON_CLASSES[i] = 1
    elif v in vocab.cons:
        VOW_CON_CLASSES[i] = 2
    else:
        raise Exception

IPA_FT_CLASSES = None
classes = set()
for v in vocab.vocab:
    if v in vocab.vows:
        classes.add(vocab.vows[v][1])
    elif v in vocab.cons:
        classes.add(vocab.cons[v][1])
# classes.add('boundary')
classes = list(classes)
class_dict = {c:i for i, c in enumerate(classes)}
class_of = np.zeros(vocab.size, dtype=np.int32)
for v in vocab.vocab:
    i = vocab.vocab2id[v]
    if v in ['#', '|', '-', '(', ')']: 
        class_of[i] = len(classes)+1
    elif v in vocab.vows:
        class_of[i] = class_dict[vocab.vows[v][1]]
    elif v in vocab.cons:
        class_of[i] = class_dict[vocab.cons[v][1]]
    else:
        raise Exception
IPA_FT_CLASSES = class_of

IDENTITY_CLASSES = np.array(range(vocab.size), dtype=np.int32)

NATURAL_CLASSES = [DEGEN_CLASSES, VOW_CON_CLASSES, IPA_FT_CLASSES, IDENTITY_CLASSES]
