import os
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse

from client_cluster import ClientCluster

# python -u run_client.py --port 8765 --data_dir /home/mesl/data --model_dir /home/mesl/model -g 3
# port should match the one in run_server.
def parse_args():
    # default setting is for extrasensory
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--data_dir', type=str, default="/home/mesl/data")
    parser.add_argument('--model_dir', type=str, default="/home/mesl/model")
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="use cuda or cpu")
    parser.add_argument('-g', '--gpu', type=int, default="6", help="gpu id")
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--buffer_size', type=int, default=1048576, help="initial buffer size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate of local model")
    return parser.parse_args()

# python main.py -t es-25 --SE --KD --pos_percent 99 --neg_percent 1
if __name__ == '__main__':
    args = parse_args()
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)
    if args.device == torch.device("cuda"):
        torch.cuda.set_device(args.gpu)

    if not os.path.exists(args.data_dir):
        raise FileExistsError(f'Not Found {args.data_dir}')

    print(args)
    print(f'#### Run Client ####')
    client_cluster = ClientCluster(args.port)
    client_cluster.run(args)