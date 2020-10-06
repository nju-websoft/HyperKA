import numpy as np


def mobius_add(u, v):
    norms_u = np.sum(np.power(u, 2), -1, keepdims=True)
    norms_v = np.sum(np.power(v, 2), -1, keepdims=True)
    inner_product = np.sum(u * v, -1, keepdims=True)
    denominator = 1 + 2 * inner_product + norms_u * norms_v
    numerator = (1 + 2 * inner_product + norms_v) * u + (1 - norms_u) * v
    results = numerator / denominator
    return results


def compute_hyperbolic_distances(vectors_u, vectors_v):
    """
    Compute poincare distances between input vectors.
    Modified based on gensim code.
    vectors_u: (batch_size, dim)
    vectors_v: (batch_size, dim)
    """
    euclidean_dists = np.linalg.norm(vectors_u - vectors_v, axis=1)  # (batch_size, )
    norms_u = np.linalg.norm(vectors_u, axis=1)  # (batch_size, )
    norms_v = np.linalg.norm(vectors_v, axis=1)  # (batch_size, )
    alpha = 1 - norms_u ** 2  # (batch_size, )
    beta = 1 - norms_v ** 2  # (batch_size, )
    gamma = 1 + 2 * ((euclidean_dists ** 2) / (alpha * beta))  # (batch_size, )
    poincare_dists = np.arccosh(gamma + 1e-10)  # (batch_size, )
    return poincare_dists  # (batch_size, )
    # distance = 2 * np.arctanh(np.linalg.norm(mobius_add(vectors_u, vectors_v), axis=1))
    # return distance


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
    # return (-dis_mat)
    # return np.exp(-dis_mat)  # sim mat
    return normalization(-dis_mat)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
