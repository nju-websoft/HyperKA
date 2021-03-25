import gc
import numpy as np
import time
import ray

from hyperka.ea_funcs.utils import div_list
from hyperka.hyperbolic.metric import compute_hyperbolic_distances, normalization

g = 1000000000


@ray.remote(num_cpus=1)
def cal_rank(task, sim, top_k):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num


def eval_alignment_mul(sim_mat, top_k, nums_threads, mess=""):
    t = time.time()
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0

    tasks = div_list(np.array(range(ref_num)), nums_threads)
    results = list()
    for task in tasks:
        res = cal_rank.remote(task, sim_mat[task, :], top_k)
        results.append(res)
    for res in ray.get(results):
        mean, mrr, num = res
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)

    acc = np.array(t_num) / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))
    return acc[0]


def cal_rank_multi_embed(frags, dic, sub_embed, embed, top_k):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    sim_mat = np.matmul(sub_embed, embed.T)  # ndarray
    # print("matmul sim mat type:", type(sim_mat))
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]
        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))

    del sim_mat
    gc.collect()
    return mean, mrr, num, mean1, mrr1, num1, prec_set


@ray.remote(num_cpus=1)
def cal_rank_multi_embed_hyperbolic(frags, sub_embed, embed, top_k):
    mr = 0
    mrr = 0
    hits = np.array([0 for k in top_k])
    sim_mat = compute_hyperbolic_similarity_single(sub_embed, embed)
    results = set()
    for i in range(len(frags)):
        ref = frags[i]
        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
        results.add((ref, aligned_e))
    del sim_mat
    return mr, mrr, hits, results


def eval_alignment_hyperbolic_multi(embed1, embed2, top_k, nums_threads, mess=""):
    t = time.time()
    ref_num = embed1.shape[0]
    hits = np.array([0 for k in top_k])
    mr = 0
    mrr = 0
    total_alignment = set()

    frags = div_list(np.array(range(ref_num)), nums_threads)
    results = list()
    for frag in frags:
        res = cal_rank_multi_embed_hyperbolic.remote(frag, embed1[frag, :], embed2, top_k)
        results.append(res)

    for res in ray.get(results):
        mr1, mrr1, hits1, alignment = res
        mr += mr1
        mrr += mrr1
        hits += hits1
        total_alignment |= alignment

    assert len(total_alignment) == ref_num

    hits = hits / ref_num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)
    mr /= ref_num
    mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, hits, mr, mrr,
                                                                                 time.time() - t))
    gc.collect()
    return hits[0]


@ray.remote(num_cpus=1)
def cal_csls_neighbor_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def csls_neighbor_sim(sim_mat, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat.shape[0])), nums_threads)
    results = list()
    for task in tasks:
        res = cal_csls_neighbor_sim.remote(sim_mat[task, :], k)
        results.append(res)
    sim_values = None
    for res in ray.get(results):
        val = res
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    return sim_values


def sim_handler_hyperbolic(embed1, embed2, k, nums_threads):
    tasks = div_list(np.array(range(embed1.shape[0])), nums_threads)
    results = list()
    for task in tasks:
        res = compute_hyperbolic_similarity.remote(embed1[task, :], embed2)
        results.append(res)
    sim_lists = list()
    for res in ray.get(results):
        sim_lists.append(res)
    sim_mat = np.concatenate(sim_lists, axis=0)
    if k == 0:
        return sim_mat
    csls1 = csls_neighbor_sim(sim_mat, k, nums_threads)
    csls2 = csls_neighbor_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat


@ray.remote(num_cpus=1)
def compute_hyperbolic_similarity(embeds1, embeds2):
    x1, y1 = embeds1.shape  # <class 'numpy.ndarray'>
    x2, y2 = embeds2.shape
    assert y1 == y2
    dist_vec_list = list()
    for i in range(x1):
        embed1 = embeds1[i, ]  # <class 'numpy.ndarray'> (y1,)
        embed1 = np.reshape(embed1, (1, y1))  # (1, y1)
        embed1 = np.repeat(embed1, x2, axis=0)  # (x2, y1)
        dist_vec = compute_hyperbolic_distances(embed1, embeds2)
        dist_vec_list.append(dist_vec)
    dis_mat = np.row_stack(dist_vec_list)  # (x1, x2)
    return normalization(-dis_mat)


def compute_hyperbolic_similarity_single(embeds1, embeds2):
    x1, y1 = embeds1.shape  # <class 'numpy.ndarray'>
    x2, y2 = embeds2.shape
    assert y1 == y2
    dist_vec_list = list()
    for i in range(x1):
        embed1 = embeds1[i, ]  # <class 'numpy.ndarray'> (y1,)
        embed1 = np.reshape(embed1, (1, y1))  # (1, y1)
        embed1 = np.repeat(embed1, x2, axis=0)  # (x2, y1)
        dist_vec = compute_hyperbolic_distances(embed1, embeds2)
        dist_vec_list.append(dist_vec)
    dis_mat = np.row_stack(dist_vec_list)  # (x1, x2)
    return normalization(-dis_mat)