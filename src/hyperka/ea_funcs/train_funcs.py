import gc
import ray
import numpy as np
import random
from sklearn import preprocessing

from hyperka.ea_apps.util import gen_adj
import hyperka.ea_funcs.utils as ut
from hyperka.hyperbolic.metric import compute_hyperbolic_similarity
from hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic

g = 1000000000


def enhance_triples(kg1, kg2, ents1, ents2):
    assert len(ents1) == len(ents2)
    print("before enhanced:", len(kg1.triples), len(kg2.triples))
    enhanced_triples1, enhanced_triples2 = set(), set()
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1.triples:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t2 is not None and t2 not in kg2.out_related_ents_dict.get(h2, set()):
            enhanced_triples2.add((h2, r1, t2))
        # if h2 is not None:
        #     enhanced_triples2.add((h2, r1, t1))
        # if t2 is not None:
        #     enhanced_triples2.add((h1, r1, t2))

    for h2, r2, t2 in kg2.triples:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t1 is not None and t1 not in kg1.out_related_ents_dict.get(h1, set()):
            enhanced_triples1.add((h1, r2, t1))
        # if h1 is not None:
        #     enhanced_triples1.add((h1, r2, t2))
        # if t1 is not None:
        #     enhanced_triples1.add((h2, r2, t1))

    print("after enhanced:", len(enhanced_triples1), len(enhanced_triples2))
    return enhanced_triples1, enhanced_triples2


def remove_unlinked_triples(triples, linked_ents):
    print("before removing unlinked triples:", len(triples))
    new_triples = set()
    for h, r, t in triples:
        if h in linked_ents and t in linked_ents:
            new_triples.add((h, r, t))
    print("after removing unlinked triples:", len(new_triples))
    return list(new_triples)


def get_transe_model(folder, kge_model, params):
    print("data folder:", folder)
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_input
    ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, _, ent_n, rel_n = read_func(folder)
    model = kge_model(ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2,
                      ori_triples1.ent_list, ori_triples2.ent_list, params)
    return ori_triples1, ori_triples2, model


def get_model(folder, kge_model, params):
    print("data folder:", folder)
    if "15" in folder:
        read_func = ut.read_dbp15k_input
    else:
        read_func = ut.read_input
    ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, _, ent_n, rel_n = read_func(folder)
    linked_entities = set(seed_sup_ent1 + seed_sup_ent2 + ref_ent1 + ref_ent2)
    enhanced_triples1, enhanced_triples2 = enhance_triples(ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2)
    triples = remove_unlinked_triples(ori_triples1.triple_list + ori_triples2.triple_list +
                                      list(enhanced_triples1) + list(enhanced_triples2), linked_entities)
    adj = gen_adj(ent_n, triples)
    model = kge_model(ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2,
                      ori_triples1.ent_list, ori_triples2.ent_list, adj, params)
    return ori_triples1, ori_triples2, model


@ray.remote(num_cpus=1)
def find_neighbours(sub_ent_list, ent_list, sub_ent_embed, ent_embed, k, metric):
    dic = dict()
    if metric == 'euclidean':
        sim_mat = np.matmul(sub_ent_embed, ent_embed.T)
        for i in range(sim_mat.shape[0]):
            sort_index = np.argpartition(-sim_mat[i, :], k + 1)
            dic[sub_ent_list[i]] = ent_list[sort_index[0:k + 1]].tolist()
    else:
        sim_mat = compute_hyperbolic_similarity(sub_ent_embed, ent_embed)
        for i in range(sim_mat.shape[0]):
            sort_index = np.argpartition(-sim_mat[i, :], k + 1)
            dic[sub_ent_list[i]] = ent_list[sort_index[0:k + 1]].tolist()
    del sim_mat
    return dic


def find_neighbours_multi_4link_(embed1, embed2, ent_list1, ent_list2, k, params, metric='euclidean'):
    if metric == 'euclidean':
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    sub_ent_list1 = ut.div_list(np.array(ent_list1), params.nums_threads)
    sub_ent_embed_indexes = ut.div_list(np.array(range(len(ent_list1))), params.nums_threads)
    results = list()
    for i in range(len(sub_ent_list1)):
        res = find_neighbours.remote(sub_ent_list1[i], np.array(ent_list2),
                                     embed1[sub_ent_embed_indexes[i], :], embed2, k, metric)
        results.append(res)
    dic = dict()
    for res in ray.get(results):
        dic = ut.merge_dic(dic, res)
    gc.collect()
    return dic


@ray.remote(num_cpus=1)
def find_neighbours_from_sim_mat(ent_list_x, ent_list_y, sim_mat, k):
    dic = dict()
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[ent_list_x[i]] = ent_list_y[sort_index[0:k + 1]].tolist()
    return dic


def find_neighbours_multi_4link_from_sim(ent_list_x, ent_list_y, sim_mat, k, nums_threads):
    ent_list_x_tasks = ut.div_list(np.array(ent_list_x), nums_threads)
    ent_list_x_indexes = ut.div_list(np.array(range(len(ent_list_x))), nums_threads)
    dic = dict()
    rest = []
    for i in range(len(ent_list_x_tasks)):
        res = find_neighbours_from_sim_mat.remote(ent_list_x_tasks[i], np.array(ent_list_y),
                                                  sim_mat[ent_list_x_indexes[i], :], k)
        rest.append(res)
    for res in ray.get(rest):
        dic = ut.merge_dic(dic, res)
    return dic


def find_neighbours_multi_4link(embed1, embed2, ent_list1, ent_list2, k, params, metric='euclidean', is_one=False):
    if metric == 'euclidean':
        sim_mat = np.matmul(embed1, embed2.T)
    else:
        sim_mat = sim_handler_hyperbolic(embed1, embed2, 0, params.nums_neg)
    neighbors1 = find_neighbours_multi_4link_from_sim(ent_list1, ent_list2, sim_mat, k, params.nums_neg)
    if is_one:
        return neighbors1, None
    neighbors2 = find_neighbours_multi_4link_from_sim(ent_list2, ent_list1, sim_mat.T, k, params.nums_neg)
    return neighbors1, neighbors2


def find_neighbours_multi(embed, ent_list, k, nums_threads, metric='euclidean'):
    if nums_threads > 1:
        ent_frags = ut.div_list(np.array(ent_list), nums_threads)
        ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), nums_threads)
        dic = dict()
        rest = []
        for i in range(len(ent_frags)):
            res = find_neighbours.remote(ent_frags[i], np.array(ent_list), embed[ent_frag_indexes[i], :], embed,
                                         k, metric)
            rest.append(res)
        for res in ray.get(rest):
            dic = ut.merge_dic(dic, res)
    else:
        dic = find_neighbours(np.array(ent_list), np.array(ent_list), embed, embed, k, metric)
    del embed
    gc.collect()
    return dic


def trunc_sampling(pos_triples, all_triples, dic, ent_list):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                candidates = dic.get(h, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                h2 = candidates[index]
            elif choice >= 500:
                candidates = dic.get(t, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                t2 = candidates[index]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples


def trunc_sampling_multi(pos_triples, all_triples, dic, ent_list, multi):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = dic.get(h, ent_list)
            h2s = random.sample(candidates, multi)
            temp_neg_triples = [(h2, r, t) for h2 in h2s]
            neg_triples.extend(temp_neg_triples)
        elif choice >= 500:
            candidates = dic.get(t, ent_list)
            t2s = random.sample(candidates, multi)
            temp_neg_triples = [(h, r, t2) for t2 in t2s]
            neg_triples.extend(temp_neg_triples)
    # neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_batch_via_neighbour(triples1, triples2, step, batch_size, neighbours_dic1, neighbours_dic2, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    neg_triples.extend(trunc_sampling_multi(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list, multi))
    neg_triples.extend(trunc_sampling_multi(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def generate_neg_triples(pos_triples, triples_data):
    all_triples = triples_data.triples
    entities = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                h2 = random.sample(entities, 1)[0]
            elif choice >= 500:
                t2 = random.sample(entities, 1)[0]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    assert len(neg_triples) == len(pos_triples)
    return neg_triples


def generate_neg_triples_multi(pos_triples, triples_data, multi):
    all_triples = triples_data.triples
    entities = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            h2s = random.sample(entities, multi)
            neg_triples.extend([(h2, r, t) for h2 in h2s])
        elif choice >= 500:
            t2s = random.sample(entities, multi)
            neg_triples.extend([(h, r, t2) for t2 in t2s])
    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_pos_neg_batch(triples1, triples2, step, batch_size, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    # for i in range(multi):
    #     choice = random.randint(0, 999)
    #     if choice < 500:
    #         h = True
    #     else:
    #         h = False
    #     # neg_triples.extend(generate_neg_triples_batch(pos_triples1, triples1, h))
    #     # neg_triples.extend(generate_neg_triples_batch(pos_triples2, triples2, h))
    #     neg_triples.extend(generate_neg_triples(pos_triples1, triples1))
    #     neg_triples.extend(generate_neg_triples(pos_triples2, triples2))
    neg_triples.extend(generate_neg_triples_multi(pos_triples1, triples1, multi))
    neg_triples.extend(generate_neg_triples_multi(pos_triples2, triples2, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_triples_of_latent_entities(triples1, triples2, entities1, entites2):
    assert len(entities1) == len(entites2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(entities1)):
        newly_triples1.extend(generate_newly_triples(entities1[i], entites2[i], triples1.rt_dict, triples1.hr_dict))
        newly_triples2.extend(generate_newly_triples(entites2[i], entities1[i], triples2.rt_dict, triples2.hr_dict))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples
