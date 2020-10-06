import tensorflow as tf
import numpy as np
from hyperka.hyperbolic.euclidean import EuclideanManifold
from hyperka.hyperbolic.util import tf_norm, tf_tanh, tf_atanh


class PoincareManifold(EuclideanManifold):
    name = "poincare"

    def __init__(self, eps=1e-15, projection_eps=1e-5, radius=1.0, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.projection_eps = projection_eps
        self.radius = radius  # the radius of the poincare ball
        self.max_norm = 1 - eps
        self.min_norm = eps

    def distance(self, u, v):
        sq_u_norm = tf.reduce_sum(u * u, axis=-1, keepdims=True)
        sq_v_norm = tf.reduce_sum(v * v, axis=-1, keepdims=True)
        sq_u_norm = tf.clip_by_value(sq_u_norm, clip_value_min=0.0, clip_value_max=self.max_norm)
        sq_v_norm = tf.clip_by_value(sq_v_norm, clip_value_min=0.0, clip_value_max=self.max_norm)
        sq_dist = tf.reduce_sum(tf.pow(u - v, 2), axis=-1, keepdims=True)
        distance = tf.acosh(1 + self.eps + (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm)) * 2))
        return distance

    def mobius_addition(self, vectors_u, vectors_v):
        norms_u = self.radius * tf.reduce_sum(tf.square(vectors_u), -1, keepdims=True)
        norms_v = self.radius * tf.reduce_sum(tf.square(vectors_v), -1, keepdims=True)
        inner_product = self.radius * tf.reduce_sum(vectors_u * vectors_v, -1, keepdims=True)
        denominator = 1 + 2 * inner_product + norms_u * norms_v
        numerator = (1 + 2 * inner_product + norms_v) * vectors_u + (1 - norms_u) * vectors_v
        denominator = tf.maximum(denominator, self.min_norm)
        results = tf.math.divide(numerator, denominator)
        # print('mobius addition', results.shape)
        return results

    def mobius_matmul(self, vectors, matrix, bias=None):
        vectors = vectors + self.eps
        matrix_ = tf.matmul(vectors, matrix) + self.eps
        matrix_norm = tf_norm(matrix_)
        vector_norm = tf_norm(vectors)
        result = 1. / np.sqrt(self.radius) * tf_tanh(
            matrix_norm / vector_norm * tf_atanh(np.sqrt(self.radius) * vector_norm)) / matrix_norm * matrix_
        if bias is None:
            return self.hyperbolic_projection(result)
        else:
            return self.hyperbolic_projection(self.mobius_addition(result, bias))

    def log_map_zero(self, vectors):
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        # vectors = vectors * 1. / np.sqrt(self.radius) * tf_atanh(np.sqrt(self.radius) * vectors_norm) / vectors_norm
        # return vectors
        diff = vectors + self.eps
        norm_diff = tf_norm(diff)
        return 1.0 / np.sqrt(self.radius) * tf_atanh(np.sqrt(self.radius) * norm_diff) / norm_diff * diff

    def exp_map_zero(self, vectors):
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        diff = vectors + self.eps
        vectors_norm = tf_norm(diff)
        vectors = tf_tanh(np.sqrt(self.radius) * vectors_norm) * vectors / (np.sqrt(self.radius) * vectors_norm)
        return vectors

    def hyperbolic_projection(self, vectors):
        # Projection operation. Need to make sure hyperbolic embeddings are inside the unit ball.
        # vectors_norm = tf.maximum(tf_norm(vectors), self.min_norm)
        # max_norm = self.max_norm / np.sqrt(self.radius)
        # cond = tf.squeeze(vectors_norm > max_norm)
        # projected = vectors / vectors_norm * max_norm
        # return tf.where(cond, projected, vectors)
        return tf.clip_by_norm(t=vectors, clip_norm=self.max_norm / np.sqrt(self.radius), axes=[-1])

    def squre_distance(self, u, v):
        distance = tf_atanh(np.sqrt(self.radius) * tf_norm(self.mobius_addition(-u, v)))
        distance = distance * 2 / np.sqrt(self.radius)
        return distance
