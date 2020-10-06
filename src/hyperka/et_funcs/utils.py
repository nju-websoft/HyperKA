import numpy as np
import time

from hyperka.et_funcs.triples import Triples


def get_ents(triples):
    heads = set([triple[0] for triple in triples])
    tails = set([triple[2] for triple in triples])
    props = set([triple[1] for triple in triples])
    ents = heads | tails
    return ents, props


def get_ids_triple(triples, ids, rel_ids):
    triple_set = set()
    for item in triples:
        triple_set.add((ids[item[0]], rel_ids[item[1]], ids[item[2]]))
    return triple_set


def sort_elements(triples, elements_set, props):
    dic = dict()
    props_dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in props:
            props_dic[p] = props_dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]

    props_sorted_list = sorted(props_dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    props_ordered_elements = [x[0] for x in props_sorted_list]
    return ordered_elements, dic, props_ordered_elements, props_dic


def generate_mapping_id(triples, ents, props, ordered=True):
    ids = dict()
    rel_ids = dict()
    if ordered:
        ordered_elements, _, props_ordered_elements, _ = sort_elements(triples, ents, props)
        for i in range(len(ordered_elements)):
            ids[ordered_elements[i]] = i
        for i in range(len(props_ordered_elements)):
            rel_ids[props_ordered_elements[i]] = i
    else:
        index = 0
        for ent in ents:
            if ent not in ids:
                ids[ent] = index
                index += 1
        prop_index = 0
        for prop in props:
            if prop not in rel_ids:
                rel_ids[prop] = prop_index
                prop_index += 1
    assert len(ids) == len(set(ents))
    return ids, rel_ids


def get_input(all_file, train_file, test_file, if_cross=False, ins_ids=None, onto_ids=None):
    if if_cross:
        print("read entity types...")
        all_triples = read_triples(all_file)
        print("# all entity types:", len(all_triples))

        train_triples = read_triples(train_file)
        print("# all train entity types:", len(train_triples))
        train_heads_id = list()
        train_tails_id = list()

        training_inst_multi = set()
        training_ins = set()
        for triple in train_triples:
            # filter the entities that not have triples in the KG
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            if triple[0] in training_inst_multi:
                continue
            training_ins.add(triple[0])
            train_heads_id.append(ins_ids[triple[0]])
            train_tails_id.append(onto_ids[triple[2]])
            training_inst_multi.add(triple[0])
        print("# selected train entity types:", len(train_heads_id))

        test_triples = read_triples(test_file)
        print("# all test entity types:", len(test_triples))

        test_heads_id_dict = dict()
        test_ins = set()
        for triple in test_triples:
            # filter the entities that not have triples in the KG
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            if ins_ids[triple[0]] not in test_heads_id_dict.keys():
                test_heads_id_dict[ins_ids[triple[0]]] = set()
            test_ins.add(triple[0])
            test_heads_id_dict[ins_ids[triple[0]]].add(onto_ids[triple[2]])

        print("training & test ins:", len(training_ins & test_ins))

        test_heads_id = list()
        test_tails_id = list()
        # ***************************************
        test_head_tails_id = list()
        test_inst_set = set()
        for triple in test_triples:
            # filter the entities that not have triples in the KG
            if triple[0] not in ins_ids.keys() or triple[2] not in onto_ids.keys():
                continue
            if triple[0] in test_inst_set:
                continue
            # filter the instances in training data
            if triple[0] in training_ins:
                continue
            test_inst_set.add(triple[0])
            test_heads_id.append(ins_ids[triple[0]])
            test_tails_id.append(onto_ids[triple[2]])
            test_head_tails_id.append(list(test_heads_id_dict[ins_ids[triple[0]]]))

        return [[train_heads_id, train_tails_id], [test_heads_id, test_tails_id, test_head_tails_id]]
    else:
        print("read KG triples...")
        if "insnet" in all_file:
            graph_name = "instance"
        else:
            graph_name = "ontology"
        triples = read_triples(all_file)
        print("all triples:", len(triples))
        ents, props = get_ents(triples)
        ids, rel_ids = generate_mapping_id(triples, ents, props)
        ids_triples = get_ids_triple(triples, ids, rel_ids)
        triples = Triples(ids_triples)

        total_ents_num = len(triples.ents)
        total_triples_num = len(triples.triple_list)
        total_props_num = len(triples.props)
        print("total " + graph_name + " ents:", total_ents_num)
        print("total " + graph_name + " props:", total_props_num)
        print("total " + graph_name + " triples:", total_triples_num)

        test_triples = read_triples(test_file)
        test_ids_triples = get_ids_triple(test_triples, ids, rel_ids)

        train_triples = read_triples(train_file)
        train_ids_triples = get_ids_triple(train_triples, ids, rel_ids)

        return [triples, train_ids_triples, test_ids_triples, total_ents_num,
                total_props_num, total_triples_num], ids


def read_input(folder):
    if "yago" not in folder:
        insnet, ins_ids = get_input(folder + "db_insnet.txt", folder + "db_insnet_train.txt",
                                    folder + "db_insnet_test.txt")
        onto, onto_ids = get_input(folder + "db_onto_small_mini.txt", folder + "db_onto_small_train.txt",
                                   folder + "db_onto_small_test.txt")
        instype = get_input(folder + "db_InsType_mini.txt", folder + "db_InsType_train.txt",
                            folder + "db_InsType_test.txt", True, ins_ids, onto_ids)
    else:
        insnet, ins_ids = get_input(folder + "yago_insnet_mini.txt", folder + "yago_insnet_train.txt",
                                    folder + "yago_insnet_test.txt")
        onto, onto_ids = get_input(folder + "yago_ontonet.txt", folder + "yago_ontonet_train.txt",
                                   folder + "yago_ontonet_test.txt")
        instype = get_input(folder + "yago_InsType_mini.txt", folder + "yago_InsType_train.txt",
                            folder + "yago_InsType_test.txt", True, ins_ids, onto_ids)
    return insnet, onto, instype


def pair2file(file, pairs):
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 3
            h = params[0]
            r = params[1]
            t = params[2]
            triples.add((h, r, t))
        f.close()
    return triples


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_adjacency_mat(triples1, triples2, ent_num, sup_ents):
    adj_mat = np.mat(np.zeros((ent_num, len(sup_ents)), dtype=np.int32))
    ht_set = triples2ht_set(triples1) | triples2ht_set(triples2)
    for i in range(ent_num):
        for j in sup_ents:
            if (i, j) in ht_set:
                adj_mat[i, sup_ents.index(j)] = 1
    print("shape of adj_mat: {}".format(adj_mat.shape))
    print("the number of 1 in adjacency matrix: {}".format(np.count_nonzero(adj_mat)))
    return adj_mat


def generate_adj_input_mat(adj_mat, d):
    w = np.random.randn(adj_mat.shape[1], d)
    m = np.matmul(adj_mat, w)
    print("shape of input adj_mat: {}".format(m.shape))
    return m


def generate_ent_attrs_sum(ent_num, ent_attrs1, ent_attrs2, attr_embeddings):
    t1 = time.time()
    ent_attrs_embeddings = None
    for i in range(ent_num):
        attrs_index = list(ent_attrs1.get(i, set()) | ent_attrs2.get(i, set()))
        assert len(attrs_index) > 0
        attrs_embeds = np.sum(attr_embeddings[attrs_index,], axis=0)
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings
