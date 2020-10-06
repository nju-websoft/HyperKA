import argparse
import ast

from hyperka.et_apps.model import HyperKA
from hyperka.et_funcs.train_funcs import get_model, train_k_epochs

parser = argparse.ArgumentParser(description='HyperKE4TI')
parser.add_argument('--input', type=str, default='../../../dataset/joie/yago/')  # db
parser.add_argument('--output', type=str, default='../../../output/results/')

parser.add_argument('--dim', type=int, default=75)
parser.add_argument('--onto_dim', type=int, default=15)
parser.add_argument('--ins_layer_num', type=int, default=3)
parser.add_argument('--onto_layer_num', type=int, default=3)
parser.add_argument('--neg_typing_margin', type=float, default=0.1)
parser.add_argument('--neg_triple_margin', type=float, default=0.2)

parser.add_argument('--nums_neg', type=int, default=20)
parser.add_argument('--mapping_neg_nums', type=int, default=20)

parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--epsilon4triple', type=float, default=1.0)
parser.add_argument('--mapping', type=bool, default=True)
parser.add_argument('--combine', type=ast.literal_eval, default=True)
parser.add_argument('--ent_top_k', type=list, default=[1, 3, 5, 10])
parser.add_argument('--nums_threads', type=int, default=8)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    triples1, triples2, model = get_model(args.input, HyperKA, args)
    iterations = 5
    trunc_ent_num1 = int(len(model.ins_entities) * (1.0 - args.epsilon4triple))
    trunc_ent_num2 = int(len(model.onto_entities) * (1.0 - args.epsilon4triple))
    print("trunc ent num for triples:", trunc_ent_num1, trunc_ent_num2)
    for iteration in range(1, args.epochs // iterations + 1):
        print("iteration ", iteration)
        train_k_epochs(model, triples1, triples2, iterations, args, trunc_ent_num1, trunc_ent_num2)
        h1 = model.test()
    print("stop")

