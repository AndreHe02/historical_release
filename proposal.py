# from evaluate import edit_distance, batch_edit_dist
import numpy as np
from data import vocab 
from tqdm import tqdm 
from evaluate import batch_edit_dist
from data import cogs

def one_edit_proposals(s, cog):
    chars = set()
    for word in cog:
        for x in word:
            chars.add(x)
    strings = [s]
    for i in range(len(s)):
        new_s = s[:i] + s[i+1:]
        strings.append(new_s)
    for i in range(len(s)):
        for x in chars:
            if x == '-' or x == '#':
                continue
            if x != s[i]:
                new_s = s[:i] + x + s[i+1:]
                strings.append(new_s)
    for i in range(len(s) + 1):
        for x in chars:
            if x == '-' or x == '#':
                continue
            new_s = s[:i] + x + s[i:]
            strings.append(new_s)
    return strings


def minimum_edit_DAG(s1, s2):

    backptrs = [[[] for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]

    mem = np.zeros((len(s1)+1, len(s2)+1), dtype=np.int32)
    mem[len(s1), len(s2)] = 0

    for i1 in range(len(s1)-1, -1, -1):
        mem[i1, len(s2)] = mem[i1+1, len(s2)] + 1
        backptrs[i1][len(s2)].append((i1+1, len(s2)))
    
    for i2 in range(len(s2)-1, -1, -1):
        mem[len(s1), i2] = mem[len(s1), i2+1] + 1
        backptrs[len(s1)][i2].append((len(s1), i2+1))
            
        for i1 in range(len(s1)-1, -1, -1):
            sub_cost = (s1[i1] != s2[i2])
            costs = (sub_cost + mem[i1+1, i2+1],
                        1 + mem[i1, i2+1],
                        1 + mem[i1+1, i2])
            min_cost = np.amin(costs, axis=0)
            mem[i1, i2] = min_cost

            if costs[0] == min_cost:
                backptrs[i1][i2].append((i1+1, i2+1))
            if costs[1] == min_cost:
                backptrs[i1][i2].append((i1, i2+1))
            if costs[2] == min_cost:
                backptrs[i1][i2].append((i1+1, i2))               
    
    return backptrs

def intermediate_strings(s1, s2, backptrs):
    strings = set()
    visited = [[False for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]

    def traverse(i1, i2, strs_on_path):
        if visited[i1][i2]:
            return 
        visited[i1][i2] = True 
        next_states = backptrs[i1][i2]
        for new_i1, new_i2 in next_states:
            if new_i1 == i1+1:
                if new_i2 == i2+1:
                    if s1[i1] != s2[i2]:
                        edit = ('sub', i1, s2[i2])
                    else:
                        traverse(new_i1, new_i2, strs_on_path)
                        continue
                else:
                    edit = ('del', i1, None)
            else:
                edit = ('ins', i1, s2[i2])

            new_strs = []
            for s in strs_on_path:
                new_strs.append(apply_edit(s, edit))
            traverse(new_i1, new_i2, strs_on_path + new_strs)
            strings.update(new_strs)

    traverse(0, 0, [s1])
    strings.add(s1)
    strings.add(s2)
    return strings 

def apply_edit(s, edit):
    type, index, char = edit
    s = str(s)  # edit a copy
    if type == 'sub':
        return s[:index] + char + s[index+1:]
    elif type == 'del':
        return s[:index] + s[index+1:]
    elif type == 'ins':
        return s[:index] + char + s[index:]

def generate_proposals_between(s1, s2):
    backptrs = minimum_edit_DAG(s1, s2)
    proposals = intermediate_strings(s1, s2, backptrs)
    return proposals 

def heuristic_proposals(s0, strings):
    proposals = set()
    for s in strings:
        proposals.update(generate_proposals_between(s0, s))
    return list(proposals)


def total_distance(proposals, strings):
    proposals_ = []
    for p in proposals:
        for _ in range(len(strings)):
            proposals_.append(p)
    strings_ = []
    for _ in range(len(proposals)):
        strings_.extend(strings)
    dists = batch_edit_dist(proposals_, strings_)
    dists = dists.reshape(len(proposals), len(strings))
    total_dists = dists.sum(axis=1)
    return total_dists 
    

def minimum_total_distance(cog):
    cur = np.random.choice(cog)
    char_set = set()
    for word in cog:
        for x in word:
            char_set.add(x)
    
    for i in range(10):
        proposals = one_edit_proposals(cur, char_set)
        dists = total_distance(proposals, cog)
        cur = proposals[np.argmin(dists)]
        cur_dist = dists[np.argmin(dists)]
    return cur, cur_dist

def evaluate_centroids():
    from data import es,it,pt,fr,la
    from tqdm import tqdm 
    centroids = []
    for i in range(len(la)):
        centroid, dist = minimum_total_distance([es[i], it[i], pt[i], fr[i]])
        centroids.append(centroid)
        print(centroid)
    print(batch_edit_dist(centroids, la).mean())
    # print(sum(centroid_dist) / len(centroid_dist))


if __name__ == "__main__":
    evaluate_centroids()

# def generate_proposals(cog, initial, max_dist):
#     cur = initial
#     char_set = set()
#     for word in cog:
#         for x in word:
#             char_set.add(x)
    
#     proposals = set()

#     def search(cur):
#         if cur in proposals:
#             return 
#         proposals.add(cur)
#         adj = one_edit_proposals(cur, char_set)
#         dists = total_distance(adj, cog)
#         for neighbor, dist in zip(adj, dists):
#             if dist <= max_dist:
#                 search(neighbor)
    
#     search(cur)
#     return proposals

    
#     # print(minimum_total_distance())


# if __name__ == '__main__':
#     cog = cogs[5]
#     print(cog)
#     centroid, dist = minimum_total_distance(cog)
#     props = generate_proposals(cog, centroid, dist+2)
#     print(props)
