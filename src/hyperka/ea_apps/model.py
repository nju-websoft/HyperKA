import gc
import time

import tensorflow as tf
from hyperka.ea_funcs.test_funcs import sim_handler_hyperbolic, eval_alignment_mul, eval_alignment_hyperbolic_multi

from hyperka.hyperbolic.poincare import PoincareManifold
from hyperka.hyperbolic.euclidean import EuclideanManifold

g = 1024 * 1024


class HGCNLayer:
    def __init__(self,
                 adj,
                 input_dim,
                 output_dim,
                 layer_id,
                 poincare,
                 bias=True,
                 act=None):
        self.poincare = poincare
        self.bias = bias
        self.act = act
        self.adj = adj
        with tf.compat.v1.variable_scope("gcn_layer_" + str(layer_id)):
            self.weight_mat = tf.compat.v1.get_variable("gcn_weights" + str(layer_id),
                                                        shape=[input_dim, output_dim],
                                                        initializer=tf.glorot_uniform_initializer(),
                                                        dtype=tf.float64)
            if bias:
                self.bias_vec = tf.compat.v1.get_variable("gcn_bias" + str(layer_id),
                                                          shape=[1, output_dim],
                                                          initializer=tf.zeros_initializer(),
                                                          dtype=tf.float64)

    def call(self, inputs, drop_rate=0.0):
        pre_sup_tangent = self.poincare.log_map_zero(inputs)
        if drop_rate > 0.0:
            pre_sup_tangent = tf.nn.dropout(pre_sup_tangent, rate=drop_rate) * (1-drop_rate)  # not scaled up
        output = tf.matmul(pre_sup_tangent, self.weight_mat)
        output = tf.sparse.sparse_dense_matmul(self.adj, output)
        output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
        if self.bias:
            bias_vec = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.bias_vec))
            output = self.poincare.mobius_addition(output, bias_vec)
            output = self.poincare.hyperbolic_projection(output)
        if self.act is not None:
            output = self.act(self.poincare.log_map_zero(output))
            output = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(output))
        return output


class HyperKA:
    def __init__(self, ent_num, rel_num, sup_ent1, sup_ent2,
                 ref_ent1, ref_ent2, kb1_entities, kb2_entities, adj, params, ):
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.sup_ent1 = sup_ent1
        self.sup_ent2 = sup_ent2
        self.sup_links = [(sup_ent1[i], sup_ent2[i]) for i in range(len(sup_ent1))]
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.ref_links = [(ref_ent1[i], ref_ent2[i]) for i in range(len(ref_ent1))]
        self.kb1_entities = kb1_entities
        self.kb2_entities = kb2_entities
        entities = ref_ent1 + ref_ent2 + sup_ent1 + sup_ent2
        self.self_links = [(entities[i], entities[i]) for i in range(len(entities))]

        self.params = params
        self.layer_num = params.gnn_layer_num
        self.adj_mat = tf.SparseTensor(indices=adj[0], values=adj[1], dense_shape=adj[2])
        self.activation = tf.tanh
        self.layers = list()
        self.output = list()

        self.new_alignment = list()
        self.new_alignment_pairs = list()

        self.dim = params.dim

        self.poincare = PoincareManifold()
        self.euclidean = EuclideanManifold()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)

        self.lr = tf.compat.v1.placeholder(tf.float32)
        self._generate_variables()
        self._generate_mapping_graph()
        self._generate_triple_graph()

        tf.compat.v1.global_variables_initializer().run(session=self.session)

    def _generate_variables(self):
        with tf.compat.v1.variable_scope('kg' + 'embeddings'):
            self.ent_embeddings = tf.compat.v1.get_variable("ent_embeds",
                                                            shape=[self.ent_num, self.dim],
                                                            initializer=tf.glorot_uniform_initializer(),
                                                            dtype=tf.float64)
            self.rel_embeddings = tf.compat.v1.get_variable("rel_embeds",
                                                            shape=[self.rel_num, self.dim],
                                                            initializer=tf.glorot_uniform_initializer(),
                                                            dtype=tf.float64)
        with tf.compat.v1.variable_scope('mapping' + 'embeddings'):
            self.mapping_matrix = tf.compat.v1.get_variable('mapping_matrix',
                                                            dtype=tf.float64,
                                                            shape=[self.dim, self.dim],
                                                            initializer=tf.initializers.orthogonal())

    def _generate_triple_graph(self):
        self.pos_hs = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.compat.v1.placeholder(tf.int32, shape=[None])

        ent_embeddings = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ent_embeddings))
        rel_embeddings = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.rel_embeddings))

        phs_embeds = tf.nn.embedding_lookup(ent_embeddings, self.pos_hs)
        prs_embeds = tf.nn.embedding_lookup(rel_embeddings, self.pos_rs)
        pts_embeds = tf.nn.embedding_lookup(ent_embeddings, self.pos_ts)
        nhs_embeds = tf.nn.embedding_lookup(ent_embeddings, self.neg_hs)
        nrs_embeds = tf.nn.embedding_lookup(rel_embeddings, self.neg_rs)
        nts_embeds = tf.nn.embedding_lookup(ent_embeddings, self.neg_ts)

        self.triple_loss = self._generate_triple_loss(phs_embeds, prs_embeds, pts_embeds,
                                                      nhs_embeds, nrs_embeds, nts_embeds)
        self.triple_optimizer = self._generate_riemannian_optimizer(self.triple_loss, self.lr)

    def _generate_mapping_graph(self):
        self.pos_entities1 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.pos_entities2 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_entities1 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.neg_entities2 = tf.compat.v1.placeholder(tf.int32, shape=[None])

        self.new_pos_entities1 = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.new_pos_entities2 = tf.compat.v1.placeholder(tf.int32, shape=[None])

        self._graph_convolution(self.params.drop_rate)
        ent_embeddings = self.output[-1]
        if self.params.combine:
            ent_embeddings = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(ent_embeddings, self.output[0]))

        pos_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.pos_entities1)
        pos_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.pos_entities2)
        neg_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.neg_entities1)
        neg_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.neg_entities2)

        new_pos_embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.new_pos_entities1)
        new_pos_embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.new_pos_entities2)

        self.mapping_loss = self._generate_mapping_loss(pos_embeds1, pos_embeds2, neg_embeds1, neg_embeds2,
                                                        new_pos_embeds1, new_pos_embeds2)
        self.mapping_optimizer = self._generate_riemannian_optimizer(self.mapping_loss, self.lr)

    def _generate_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_distance = self.poincare.distance(self.poincare.mobius_addition(phs, prs), pts)
        neg_distance = self.poincare.distance(self.poincare.mobius_addition(nhs, nrs), nts)
        pos_score = tf.reduce_sum(pos_distance, 1)
        neg_score = tf.reduce_sum(neg_distance, 1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_triple_margin, dtype=tf.float64) - neg_score))
        return pos_loss + neg_loss

    def _generate_mapping_loss(self, pos_embeds1, pos_embeds2, neg_embeds1, neg_embeds2, new_embeds1, new_embeds2):
        mapped_sup_embeds1 = self.poincare.mobius_matmul(pos_embeds1, self.mapping_matrix)
        pos_distance = tf.norm(self.poincare.distance(mapped_sup_embeds1, pos_embeds2), axis=-1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_distance))

        mapped_neg_embeds1 = self.poincare.mobius_matmul(neg_embeds1, self.mapping_matrix)
        neg_distance = tf.norm(self.poincare.distance(mapped_neg_embeds1, neg_embeds2), axis=-1)
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.params.neg_align_margin, dtype=tf.float64) - neg_distance))

        new_mapped_sup_embeds1 = self.poincare.mobius_matmul(new_embeds1, self.mapping_matrix)
        new_pos_distance = tf.norm(self.poincare.distance(new_mapped_sup_embeds1, new_embeds2), axis=-1)
        new_pos_loss = tf.reduce_sum(tf.nn.relu(new_pos_distance))

        return pos_loss + neg_loss + self.params.bp_param * new_pos_loss

    def _graph_convolution(self, drop_rate):
        self.output = list()  # reset
        output_embeddings = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ent_embeddings))
        self.output.append(output_embeddings)
        for i in range(self.layer_num):
            activation = self.activation
            if i == self.layer_num - 1:
                activation = None
            gcn_layer = HGCNLayer(self.adj_mat, self.dim, self.dim, i, self.poincare, act=activation)
            self.layers.append(gcn_layer)
            output_embeddings = gcn_layer.call(output_embeddings, drop_rate=drop_rate)
            output_embeddings = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(output_embeddings, self.output[-1]))
            self.output.append(output_embeddings)

    def _graph_convolution_for_evaluation(self):
        output = list()
        output_embeddings = self.poincare.hyperbolic_projection(self.poincare.exp_map_zero(self.ent_embeddings))
        output.append(output_embeddings)
        for i in range(self.layer_num):
            gcn_layer = self.layers[i]
            output_embeddings = gcn_layer.call(output_embeddings)
            output_embeddings = self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(output_embeddings, output[-1]))
            output.append(output_embeddings)
        if self.params.combine:
            return self.poincare.hyperbolic_projection(
                self.poincare.mobius_addition(output_embeddings, output[0]))
        return output_embeddings

    @staticmethod
    def _generate_riemannian_optimizer(hyperbolic_loss, lr):
        opt = tf.compat.v1.train.AdamOptimizer(lr)
        trainable_grad_vars = opt.compute_gradients(hyperbolic_loss)
        grad_vars = [(g, v) for g, v in trainable_grad_vars if g is not None]
        rescaled = [(g * (1. - tf.reshape(tf.norm(v, axis=1), (-1, 1)) ** 2) ** 2 / 4., v) for g, v in grad_vars]
        train_op = opt.apply_gradients(rescaled)
        return train_op

    def test(self, k=10):
        ti = time.time()
        output_embeddings = self._graph_convolution_for_evaluation()
        refs1_embed = tf.nn.embedding_lookup(output_embeddings, self.ref_ent1)
        refs2_embed = tf.nn.embedding_lookup(output_embeddings, self.ref_ent2)
        refs1_embed = self.poincare.mobius_matmul(refs1_embed, self.mapping_matrix)
        refs1_embed = refs1_embed.eval(session=self.session)
        refs2_embed = refs2_embed.eval(session=self.session)
        if k > 0:
            mess = "ent alignment by hyperbolic and csls"
            sim = sim_handler_hyperbolic(refs1_embed, refs2_embed, k, self.params.nums_threads)
            hits1 = eval_alignment_mul(sim, self.params.ent_top_k, self.params.nums_threads, mess=mess)
        else:
            mess = "fast ent alignment by hyperbolic"
            hits1 = eval_alignment_hyperbolic_multi(refs1_embed, refs2_embed, self.params.ent_top_k,
                                                    self.params.nums_threads, mess)
        print("test totally costs {:.3f} s ".format(time.time() - ti))
        del refs1_embed, refs2_embed
        gc.collect()
        return hits1

    def eval_ent_embeddings(self):
        ent_embeddings = self._graph_convolution_for_evaluation()
        return ent_embeddings.eval(session=self.session)

    def eval_kb12_embed(self):
        ent_embeddings = self._graph_convolution_for_evaluation()
        embeds1 = tf.nn.embedding_lookup(ent_embeddings, self.kb1_entities)
        embeds2 = tf.nn.embedding_lookup(ent_embeddings, self.kb2_entities)
        return embeds1.eval(session=self.session), embeds2.eval(session=self.session)

    def eval_kb1_input_embed(self, is_map=False):
        embeds = tf.nn.embedding_lookup(self.ent_embeddings, self.kb1_entities)
        if is_map:
            embeds = self.poincare.mobius_matmul(embeds, self.mapping_matrix)
        return embeds.eval(session=self.session)

    def eval_kb2_input_embed(self):
        return tf.nn.embedding_lookup(self.ent_embeddings, self.kb2_entities).eval(session=self.session)

    def eval_output_embed(self, index, is_map=False):
        output_embeddings = self._graph_convolution_for_evaluation()
        embeds = tf.nn.embedding_lookup(output_embeddings, index)
        if is_map:
            embeds = self.poincare.mobius_matmul(embeds, self.mapping_matrix)
        return embeds.eval(session=self.session)
