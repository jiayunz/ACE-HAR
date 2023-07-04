import os
import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import argparse

from general_utils import set_seed
from server import Server
sys.path.append('utils')
from logger import Logger

def main(args):
    args.save_dir = os.path.join(args.save_dir, f"{args.task}/seed{args.seed}/")
    args.data_shares = [1 / args.total_clients for _ in range(args.total_clients)]

    if args.sample_clients is None:
        args.sample_clients = args.total_clients

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_path = os.path.join(args.save_dir, f"iid{args.iid}.{args.algorithm}.tc{args.total_clients}.sc{args.sample_clients}.log")  # + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    args.logger = Logger(file_path=log_path).get_logger()
    args.logger.critical(log_path)
    torch.cuda.set_device(args.gpu)
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    set_seed(args.seed)
    if args.task.startswith('cifar'):
        args.client_model_names = {i: 'ResNet_1layer' for i in range(args.total_clients)}
    elif args.task == 'mnli':
        args.client_model_names = {i: 'DistilBERT' for i in range(args.total_clients)}

    if args.task == 'pamap2':
        args.dataset = 'pamap2'
        args.total_clients = 9
        args.n_class = 12
        args.metrics = ['F1', 'AUC', 'ACC']
    elif args.task == 'cifar10':
        args.dataset = 'cifar10'
        args.n_class = 10
        args.metrics = ['F1', 'AUC', 'ACC']
    elif args.task == 'cifar100':
        args.dataset = 'cifar100'
        args.n_class = 100
        args.metrics = ['F1', 'AUC', 'ACC']
    elif args.task == 'mnli':
        args.dataset = 'mnli'
        args.n_class = 3
        args.epochs = 1
        args.metrics = ['F1', 'AUC', 'ACC']

    args.logger.critical(args)

    server = Server(args)
    args.logger.debug('Server created.')

    for client_id, (client_ip, client_port) in client_addr.items():
        server.register_client(client_id, client_ip, client_port)

    server.train(args)

    del args
    del server


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4321, help="random seed")
    parser.add_argument('-t', '--task', choices=['cifar10'], default='cifar10', help="task name")
    parser.add_argument('-g', '--gpu', type=int, default="7", help="gpu id")
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--save_dir', type=str, default="logs/")
    # training & communication
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="use cuda or cpu")
    parser.add_argument('--algorithm', choices=['fedavg', 'fedadam'], default='fedavg', help="algorithm for model aggregation")
    parser.add_argument('--iid', action='store_true', help="whether the data is iid or non-iid, default is non-iid")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for dirichlet distribution")
    parser.add_argument('--total_clients', type=int, default=6, help="number of total clients")
    parser.add_argument('--sample_clients', type=int, help="number of clients join training at each round")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of training epochs per round")
    parser.add_argument('-r', '--rounds', type=int, default=50, help="number of communication rounds")
    parser.add_argument('--buffer_size', type=int, default=1048576)
    parser.add_argument('--timeout', type=int, default=7200)
    # model parameter
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden dim in hypernet")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of hypernet") # important

    return parser.parse_args()


# change (addresses, ips) in client_clusters
# python -u run_server.py -t cifar10 --total_clients 2 -g 0
# total_clients can be larger than the size of client_clusters. They will be automatically distributed to the client clusters
if __name__ == '__main__':
    args = parse_args()
    if args.seed is not None:
        random_seeds = [4321, 4322, 4323, 4324, 4325]
    else:
        random_seeds = [args.seed]

    client_clusters = [
        ('137.110.160.24', 12345),
        ('137.110.160.23', 12345),
        ('137.110.160.116', 12345),
        ('137.110.160.64', 12345),
        ('137.110.160.66', 12345)
    ]
    client_addr = {i: client_clusters[i % len(client_clusters)] for i in range(args.total_clients)}

    for seed in random_seeds:
        args = parse_args()
        args.seed = seed
        main(args)
