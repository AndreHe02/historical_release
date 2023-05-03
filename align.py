from data import vocab
from models import BaseModel
import numpy as np

def align(sources, targets):
    batch_size = len(sources)
    model = BaseModel(None, False)
    
    step = vocab.vocab[vocab.step]
    backpointers, sources, source_lens, targets, target_lens = model.edit_prob(
            sources, targets, use_max_seq=True, count=False, return_backpointers=True)
    aligned_targets = ['' for _ in range(batch_size)]

    for n in range(batch_size):
        source, target = sources[:, n], targets[:, n]
        i1 = max(source_lens) - source_lens[n]
        i2 = max(target_lens) - target_lens[n]
        in_insertion = False 
        while i1 < len(source) or i2 < len(target):
            new_i1, new_i2 = backpointers[n][i1][i2][0]
            if new_i1 > i1:
                if new_i2 > i2:
                    if in_insertion:
                        aligned_targets[n] += step
                    aligned_targets[n] += vocab.vocab[target[i2]]
                    # print('sub', source[i1], target[i2])
                    in_insertion = True 
                else:
                    if in_insertion:
                        aligned_targets[n] += step
                    aligned_targets[n] += step
                    # print('del', source[i1])
                    in_insertion = False
            else:
                if not in_insertion:
                    aligned_targets[n] = aligned_targets[n][:-1]
                aligned_targets[n] += vocab.vocab[target[i2]]
                # print('ins', target[i2])
                in_insertion = True
            i1, i2 = new_i1, new_i2
        if in_insertion:
            aligned_targets[n] += step
    return aligned_targets



if __name__ == "__main__":
    cogs = CognateDataset('cognates.txt')
    vocab = Vocabulary(cogs.cognates, use_classes=1)
    es = cogs.words_in(3)
    it = cogs.words_in(2)

    sources = es
    targets = it
    sources = ['(%s' % s for s in sources]
    targets = ['(%s' % s for s in targets]
    aligned = align(sources, targets, True)


    for i, (s, t) in enumerate(zip(sources, aligned)):
        num_steps = 0
        for x in t:
            if x == '|':
                num_steps += 1
        # print(num_steps, len(s))
        print(s, t)
        assert num_steps == len(s)


#     idxs = []
#     for _ in range(20):
#         idxs.append(np.random.randint(0, len(sources)))
    
#     for i in idxs:
#         source, aligned = sources[i], aligned_targets[i]
#         print(aligned)
#         s1 = ''.join([c + '\t' for c in source])
#         print(s1)
#         s2 = ''
#         for x in aligned:
#             if x == '|':
#                 s2 += '\t'
#             else:
#                 s2 += x
#         print(s2)
#         print()