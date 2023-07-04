import torch
from torch import nn
import numpy as np
import pickle
from copy import deepcopy
from collections import defaultdict, OrderedDict
import time
import socket
import threading
from communication_utils import recv, send

from evaluation import display_results
from utils.build_model import build_model
from utils.prepare_model_weights import prepare_client_weights
from utils.model_aggregation import FedAvg, FedAdam

EPS = 1e-7

class Server():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = args.port
        self.total_clients = args.total_clients
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.device = args.device
        self.metrics = args.metrics
        self.client_clusters = defaultdict(set)
        self.client_addr = {}
        self.client_features = {}
        self.logger = args.logger
        self.iid = args.iid
        self.client_model_names = args.client_model_names

        # the first in the list is the final evaluated one
        self.client_models = {}
        for cid, model_name in args.client_model_names.items():
            self.client_models[cid] = build_model(model_name, args.task, args.n_class, args.device)
            self.logger.debug(f'Client {cid} model created')

        # create global_model_weights after activate adapters
        self.global_model_weights = OrderedDict()
        for cid, model_name in args.client_model_names.items():
            for k, p in self.client_models[cid].named_parameters():
                if not p.requires_grad:
                    continue
                if k not in self.global_model_weights:
                    self.global_model_weights[k] = p
                elif p.size() != self.global_model_weights[k].size():
                    global_k = model_name + '.' + k
                    self.global_model_weights[global_k] = p

        self.global_keys = list(self.global_model_weights.keys())
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}

        if args.algorithm == 'fedavg':
            self.aggregator = FedAvg(self.global_keys, self.client_model_names)
        elif args.algorithm == 'fedadam':
            self.aggregator = FedAdam(self.global_keys, self.client_model_names)

    def register_client(self, id, ip, port):
        self.client_addr[id] = (ip, port)
        self.client_clusters[(ip, port)].add(id)

    def server_aggregate_weighted(self, client_model_weights, client_weights):
        # client_weights: number of data samples
        global_dict = {}

        for k, p in self.global_model_weights.items():
            k_update_weights = []
            stacked_model_weights = []
            for c, model_weights in client_model_weights.items():
                if k in model_weights and p.size() == model_weights[k].size():
                    k_update_weights.append(client_weights[c])
                    stacked_model_weights.append(model_weights[k].cpu())
                elif k.startswith(self.client_model_names[c]):
                    model_k = k[len(self.client_model_names[c]) + 1:]
                    if model_k in model_weights and p.size() == model_weights[model_k].size():
                        k_update_weights.append(client_weights[c])
                        stacked_model_weights.append(model_weights[model_k].cpu())

            if not len(k_update_weights):
                self.logger.debug(f'{k} {p.size()} not found in any client model.')
                continue
            k_update_weights = torch.FloatTensor(k_update_weights)
            k_update_weights /= k_update_weights.sum()
            for _ in p.shape:
                k_update_weights = k_update_weights.unsqueeze(-1)

            stacked_model_weights = torch.stack(stacked_model_weights, 0)
            global_dict[k] = (k_update_weights * stacked_model_weights).sum(0) # todo: change self.total_clients

        self.global_model_weights.update(global_dict)


    def train(self, args):
        # types of messenge that server send to client
        # train: ask client to train model and return the model parameter
        # update: send the updated model to the client
        # stop: ask client to stop training and close connection

        self.logger.debug('---Start Registration---')
        threads = {}
        for cluster, cids in self.client_clusters.items():
            self.port = ((self.port - 1024) % (65535 - 1024)) + 1025
            send_msg = pickle.dumps({"subject": "register", "data": {"args": args, "ids": cids, "global_keys": self.global_keys}})

            socket_thread = SocketThread(
                addr=(self.ip, self.port),
                client_addr=cluster,
                send_msg=send_msg,
                buffer_size=args.buffer_size,
                timeout=self.timeout,
                logger=self.logger
            )
            socket_thread.start()
            threads[cluster] = socket_thread

        for cluster in threads:
            threads[cluster].join()
            self.client_features.update(threads[cluster].get_result()["client_features"])
        self.logger.debug('---Finish Registration---')

        for r in range(args.rounds):
            start_time = time.time()
            selected_clients = sorted(np.random.permutation(list(self.client_addr.keys()))[:args.sample_clients])
            self.logger.critical(f'selected_clients: {selected_clients}')

            threads = {}
            for cluster in self.client_clusters:
                train_clients = [c for c in selected_clients if c in self.client_clusters[cluster]]
                eval_clients = self.client_clusters[cluster] - set(train_clients)

                send_msg = {"subject": "train_and_eval", "data": {
                     "round": r,
                     "train": {'ids': train_clients, "global_model": list(self.global_model_weights.values())},
                     "eval": {"ids": eval_clients, "global_model": list(self.global_model_weights.values())}
                }}
                self.port = ((self.port - 1024) % (65535 - 1024)) + 1025

                socket_thread = SocketThread(
                    addr=(self.ip, self.port),
                    client_addr=cluster,
                    send_msg=pickle.dumps(send_msg),
                    buffer_size=args.buffer_size,
                    timeout=self.timeout,
                    logger=self.logger
                )
                # current version: all client models have the same model architecture
                socket_thread.start()
                threads[cluster] = socket_thread

            client_response = defaultdict(dict)
            for cluster in threads:
                threads[cluster].join()
                client_response.update(threads[cluster].get_result())
            update_client_weights = {c: res['model'] for c, res in client_response.items() if c in selected_clients}

            # store weights from large device
            for c in selected_clients:
                update_client_weights[c] = prepare_client_weights(self.client_models[c], self.client_model_names[c], {self.global_keys[k]: p for k, p in update_client_weights[c].items()})
                missing_keys, unexpected_keys = self.client_models[c].load_state_dict(update_client_weights[c], strict=False)
                self.logger.debug('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))

            avg_scores = {}
            for metric in self.metrics:
                avg_scores[metric] = np.average([res['score'][metric] for c, res in client_response.items()])

            torch.cuda.empty_cache()

            self.logger.debug('Model Aggregation')
            updated_model_weights = self.aggregator.update(self.global_model_weights, update_client_weights, {c: args.data_shares[c] for c in range(args.total_clients)})
            self.global_model_weights.update(updated_model_weights)

            end_time = time.time()
            duration = (end_time - start_time) / 60.
            self.logger.critical('[TRAIN] Round %i, time=%.3fmins' % (r, duration))
            display_results(avg_scores, self.metrics, self.logger)
            for c in client_response:
                self.logger.critical({c: {m: round(client_response[c]['score'][m], 4) for m in self.metrics}})


class SocketThread(threading.Thread):
    def __init__(self, addr, client_addr, send_msg, buffer_size=1024, timeout=10, logger=None):
        threading.Thread.__init__(self)
        self.addr = addr
        self.client_addr = client_addr
        self.send_msg = send_msg
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.logger = logger

    def run(self):
        try:
            self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.bind(self.addr)
            self.soc.connect(self.client_addr)
            self.logger.debug(f"Run a Thread for Connection with {self.client_addr}.")
            self.logger.debug(f"Send {len(self.send_msg) * 1e-9} Gb to client.")
            send(self.soc, self.send_msg, self.buffer_size)

            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting to Receive Data from {self.client_addr} Starting from {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec} GMT"
            self.logger.debug(date_time)
            msg, status = recv(self.soc, self.buffer_size, self.timeout)
            self.received_data = msg["data"] # model weight
            self.logger.debug(f"Receive {msg['subject'].upper()} message from {self.client_addr}")
            if status == 0:
                self.logger.debug(f"Connection Closed with {self.client_addr} either due to inactivity for {self.timeout} seconds or due to an error.")

        except BaseException as e:
            self.logger.error(f"Error Connecting to the Client {self.client_addr}: {e}")

        finally:
            self.soc.close()
            self.logger.debug(f'Close connection with {self.client_addr}.')

    def get_result(self):
        try:
            return self.received_data
        except Exception as e:
            self.logger.error(f"Error Getting Result from {self.client_addr}: {e}.")
            return None