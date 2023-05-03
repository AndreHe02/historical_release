import numpy as np
import argparse
from data import vocab, la, pt, it, es, fr

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def batch_edit_dist(sources, targets):
    batch_size = len(sources)
    sources, source_lens = vocab.make_batch(sources, align_right=True)
    targets, target_lens = vocab.make_batch(targets, align_right=True)

    i1_back = np.zeros((len(sources)+1, len(targets)+1, batch_size), dtype=np.int32)
    i2_back = np.zeros((len(sources)+1, len(targets)+1, batch_size), dtype=np.int32)
    
    mem = np.zeros((len(sources)+1, len(targets)+1, batch_size), dtype=np.int32)
    mem[len(sources), len(targets)] = 0

    for i1 in range(len(sources)-1, -1, -1):
        mem[i1, len(targets)] = mem[i1+1, len(targets)] + 1
        i1_back[i1, len(targets)] = i1+1
        i2_back[i1, len(targets)] = len(targets)
    
    for i2 in range(len(targets)-1, -1, -1):
        mem[len(sources), i2] = mem[len(sources), i2+1] + 1
        i1_back[len(sources), i2] = len(sources)
        i2_back[len(sources), i2] = i2+1
            
        for i1 in range(len(sources)-1, -1, -1):
            sub_cost = (sources[i1] != targets[i2])
            costs = (sub_cost + mem[i1+1, i2+1],
                        1 + mem[i1, i2+1],
                        1 + mem[i1+1, i2])
            min_cost = np.amin(costs, axis=0)
            mem[i1, i2] = min_cost

            choice = np.argmin(costs, axis=0)
            for n in range(batch_size):
                if choice[n] == 0:
                    i1_back[i1, i2, n] = i1+1
                    i2_back[i1, i2, n] = i2+1
                elif choice[n] == 1:
                    i1_back[i1, i2, n] = i1
                    i2_back[i1, i2, n] = i2+1       
                elif choice[n] == 2:
                    i1_back[i1, i2, n] = i1+1
                    i2_back[i1, i2, n] = i2        

    source_start = max(source_lens) - source_lens
    target_start = max(target_lens) - target_lens
    return mem[source_start, target_start, range(batch_size)]



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--source')
    # args = parser.parse_args()
    # source_file = args.source

    # targets = la

    # with open(source_file, 'r') as f:
    #     lines = f.readlines()
    #     sources = [w.strip() for w in lines]

    # dists = batch_edit_dist(sources, targets)
    # print(dists.mean())

    # print(batch_edit_dist(la, es).mean())
    # print(batch_edit_dist(la, it).mean())
    # print(batch_edit_dist(la, fr).mean())
    # print(batch_edit_dist(la, pt).mean())

    total_len = 0
    total_count = 0
    for l in [la, es, it, fr, pt]:
        for word in l:
            total_len += len(word)
            total_count += 1
    print((total_len, total_count, total_len/total_count))
    