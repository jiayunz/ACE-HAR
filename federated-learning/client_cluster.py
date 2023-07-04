import os
import torch
from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from datetime import datetime
import socket
from communication_utils import send, recv
from general_utils import set_seed
from network import adjust_net
from transformers import AutoAdapterModel
from tqdm import tqdm

from evaluation import calculate_SLC_metrics, display_results
from utils.prepare_model_weights import prepare_client_weights, convert_model_key_to_idx
import sys
sys.path.append('utils/models')
from resnet import ResNet18, ResNet_1layer
from lenet import LeNet

EPS = 1e-7

class ClientCluster():
    def __init__(self, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = port
        self.server_ip = None
        self.clients = {}
        print('address:', (self.ip, self.port))

    def register_task(self, args, server_args, global_keys):
        self.global_keys = global_keys
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}

        if server_args.task == 'pamap2':
            from dataset import load_pamap2, collate_fn
            trainData, valData, testData = load_pamap2(os.path.join(args.data_dir, 'pamap2/pamap2_data_100.pkl'), os.path.join(args.data_dir, 'pamap2/pamap2_label_100.pkl'), client_id=id, seed=server_args.seed)
            collate_fn = collate_fn

        elif server_args.task.startswith('cifar'):
            from utils.load_cifar import load_iid_cifar, load_noniid_cifar
            if server_args.iid:
                trainData, valData, testData = load_iid_cifar(server_args.dataset, os.path.join(args.data_dir, server_args.dataset), server_args.data_shares)
            else:
                trainData, valData, testData = load_noniid_cifar(server_args.dataset, os.path.join(args.data_dir, server_args.dataset), server_args.data_shares, server_args.alpha)
            collate_fn = None

        elif server_args.task == 'mnli':
            valData = [None] * server_args.total_clients
            from utils.load_mnli import load_iid_mnli, load_noniid_mnli, collate_fn
            if server_args.iid:
                trainData, testData = load_iid_mnli(os.path.join(args.data_dir, server_args.task, 'original'), server_args.data_shares)
            else:
                trainData, testData = load_noniid_mnli(os.path.join(args.data_dir, server_args.task, 'original'), server_args.data_shares, server_args.alpha)
            collate_fn = collate_fn
        else:
            raise ValueError('Wrong dataset.')

        return trainData, valData, testData, collate_fn


    def run(self, args):
        self.device = args.device
        # waiting for server to send request
        try:
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            soc.bind((self.ip, self.port))
            soc.listen(1)
            print('Start Listening...')

            while True:
                try:
                    new_socket, source_addr = soc.accept()
                    new_socket.settimeout(args.timeout)
                    if self.server_ip is not None and source_addr[0] != self.server_ip:
                        new_socket.close()
                        print(f'\033[31mReceive Unexpected Connection from {source_addr}. Connection Close.\033[0m')

                    print(f'Receive connection from {source_addr}')
                    # receive request
                    msg, status = recv(new_socket, args.buffer_size, recv_timeout=60)
                    if status == 1:
                        print(f"Receive {msg['subject'].upper()} message from {source_addr}")

                    if isinstance(msg, dict):
                        if msg['subject'] == 'register':
                            self.server_ip = source_addr[0]
                            trainData, valData, testData, collate_fn = self.register_task(args, msg['data']['args'], msg['data']['global_keys'])
                            client_features = {}
                            for cid in msg['data']['ids']:
                                self.clients[cid] = Client(args, msg['data']['args'], cid, trainData[cid], valData[cid], testData[cid], collate_fn)
                                client_features[cid] = self.clients[cid].class_distribution

                            data_byte = pickle.dumps({"subject": "register", "data": {"client_features": client_features}})
                            print("Registered. Reply to the Server.")
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte

                        elif msg['subject'] == 'evaluate':
                            response_data = {}
                            for cid in msg['data']['ids']:
                                # don't update client model
                                model = deepcopy(self.clients[cid].model)
                                new_weights = prepare_client_weights(self.clients[cid].model, self.clients[cid].model_name, msg['data']['model'][cid])
                                missing_keys, unexpected_keys = model.load_state_dict(new_weights, strict=False)
                                print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))
                                model = model.to(self.device)

                                model = self.clients[cid].fine_tune(args, model)
                                test_scores = self.clients[cid].evaluate(args, model)
                                display_results(test_scores, self.clients[cid].metrics)
                                response_data[cid] = {"score": test_scores}

                                del model

                            # reply request
                            data_byte = pickle.dumps({"subject": "evaluate", "data": response_data})
                            print(f"Evaluated. Send {len(data_byte)*1e-9} Gb to the Server.")
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte

                        elif msg['subject'] == 'train':
                            response_data = {}
                            for cid in msg['data']['ids']:
                                new_weights = prepare_client_weights(self.clients[cid].model, self.clients[cid].model_name, msg['data']['model'][cid])
                                updated_weights, test_scores = self.clients[cid].train(args, msg['data']['round'], new_weights)
                                display_results(test_scores, self.clients[cid].metrics)
                                response_data[cid] = {"model": updated_weights, "score": test_scores}

                            # reply request
                            data_byte = pickle.dumps({"subject": "train", "data": response_data})
                            print(f"Trained. Send {len(data_byte)*1e-9} Gb to the Server.")
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte

                        elif msg['subject'] == 'train_and_eval':
                            response_data = {}
                            # train
                            for cid in msg['data']['train']['ids']:
                                recv_weights = dict(zip(self.global_keys, msg['data']['train']['global_model']))
                                new_weights = prepare_client_weights(self.clients[cid].model, self.clients[cid].model_name, recv_weights)
                                updated_weights, test_scores = self.clients[cid].train(args, msg['data']['round'], new_weights)
                                display_results(test_scores, self.clients[cid].metrics)
                                response_data[cid] = {"model": convert_model_key_to_idx(self.global_key_to_idx, self.clients[cid].model_name, updated_weights), "score": test_scores}

                            # eval
                            for cid in msg['data']['eval']['ids']:
                                # don't update client model
                                model = deepcopy(self.clients[cid].model)
                                recv_weights = dict(zip(self.global_keys, msg['data']['eval']['global_model']))
                                new_weights = prepare_client_weights(self.clients[cid].model, self.clients[cid].model_name, recv_weights)

                                missing_keys, unexpected_keys = model.load_state_dict(new_weights, strict=False)
                                print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))
                                model = model.to(self.device)
                                model = self.clients[cid].fine_tune(args, model)
                                test_scores = self.clients[cid].evaluate(args, model)
                                display_results(test_scores, self.clients[cid].metrics)
                                response_data[cid] = {"score": test_scores}

                            # reply request
                            data_byte = pickle.dumps({"subject": "train_and_eval", "data": response_data})
                            print(f"Trained and evaluated. Send {len(data_byte)*1e-9} Gb to the Server.")
                            new_socket.settimeout(3600)
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte
                finally:
                    new_socket.close()
                    print(f'Close Connection with {source_addr}')
        finally:
            soc.close()

class Client():
    def __init__(self, args, server_args, id, trainData, valData, testData, collate_fn):
        self.id = id
        args.epochs = server_args.epochs
        args.buffer_size = server_args.buffer_size
        args.save_dir = os.path.join(args.model_dir, f"{server_args.task}/seed{server_args.seed}/")
        set_seed(server_args.seed)
        self.task = server_args.task
        self.device = args.device
        self.metrics = server_args.metrics

        self.trainData = trainData
        self.valData = valData
        self.testData = testData
        self.collate_fn = collate_fn

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        if self.task == 'cifar10':
            self.n_class = 10
        elif self.task == 'cifar100':
            self.n_class = 100
        elif self.task == 'mnli':
            self.n_class = 3
        else:
            raise TypeError('Wrong self.trainData type.')

        # client features
        class_distribution = np.zeros(self.n_class)
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        for _, labels in train_loader:
            for cls in range(self.n_class):
                class_distribution[cls] += labels.numpy().tolist().count(cls)

        self.class_distribution = class_distribution / np.sum(class_distribution)
        print(f'Client {id} class distribution:', self.class_distribution)
        self.model_name = server_args.client_model_names[id]
        if self.model_name == 'ResNet18':
            self.model = adjust_net(ResNet18(self.n_class))
        elif self.model_name == 'ResNet_1layer':
            self.model = adjust_net(ResNet_1layer(num_blocks=[2], num_classes=self.n_class))
        elif self.model_name == 'LeNet':
            self.model = adjust_net(LeNet(self.n_class))
        elif self.model_name == 'BERT':
            self.model = AutoAdapterModel.from_pretrained('bert-base-uncased')
            self.model.add_adapter("mnli")
            self.model.add_classification_head("mnli", num_labels=self.n_class)
            self.model.train_adapter("mnli")
        elif self.model_name == 'DistilBERT':
            self.model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
            self.model.add_adapter("mnli")
            self.model.add_classification_head("mnli", num_labels=self.n_class)
            self.model.train_adapter("mnli")

        print('save_dir:', args.save_dir)
        print(f'Client {self.id} n_train: {len(self.trainData)}, n_class: {self.n_class}')


    def local_update(self, args, r):
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        durations = []
        local_loss = []
        for e in range(args.epochs):
            start_time = datetime.now()
            for sample, label in tqdm(train_loader, total=len(train_loader)):
            #for step, (sample, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
                avg_sample_loss = self.train_one_batch(sample, label, optimizer)
                local_loss.append(avg_sample_loss)

            end_time = datetime.now()
            duration = (end_time - start_time).seconds / 60.
            durations.append(duration)
            print('[TRAIN] Client %i, Epoch %i, time=%.3fmins' % (self.id, r * args.epochs + e, duration))

        print('avg_time=%.3fmins' % np.average(durations))
        return np.average(local_loss), np.std(local_loss)

    def train_one_batch(self, sample, label, optimizer):
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        label = label.to(self.device, dtype=torch.long)
        if len(label.shape) > 1:
            label = torch.argmax(label, dim=-1)

        optimizer.zero_grad()
        if self.task == 'mnli':
            out = self.model(sample[0].to(self.device), token_type_ids=sample[1].to(self.device), attention_mask=sample[2].to(self.device))['logits']
        else:
            out = self.model(sample.to(self.device))

        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        return loss.item()

    # train model one round, without changing self.model value
    def fine_tune(self, args, model):
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        model = model.to(self.device)
        model.train()

        for e in range(args.epochs):
            start_time = datetime.now()
            for sample, label in tqdm(train_loader, total=len(train_loader)):
                criterion = nn.CrossEntropyLoss()

                label = label.to(self.device, dtype=torch.long)
                if len(label.shape) > 1:
                    label = torch.argmax(label, dim=-1)

                optimizer.zero_grad()
                if self.task == 'mnli':
                    out = model(sample[0].to(self.device), token_type_ids=sample[1].to(self.device), attention_mask=sample[2].to(self.device))['logits']
                else:
                    out = model(sample.to(self.device))

                loss = criterion(out, label)
                loss.backward()
                optimizer.step()

            end_time = datetime.now()
            duration = (end_time - start_time).seconds / 60.
            print('[FINE-TUNE] Client %i, time=%.3fmins' % (self.id, duration))

        return model

    def evaluate(self, args, model):
        data_loader = DataLoader(self.testData, batch_size=args.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)

        y_pred = []
        y_true = []

        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            for sample, label in data_loader:
                label = label.to(self.device, dtype=torch.float)
                if len(label.shape) == 1:
                    label = F.one_hot(label.to(torch.long), num_classes=self.n_class)

                if self.task == 'mnli':
                    out = model(sample[0].to(self.device), token_type_ids=sample[1].to(self.device), attention_mask=sample[2].to(self.device))['logits']
                else:
                    out = model(sample.to(self.device))
                out = torch.softmax(out, dim=-1)

                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_scores = calculate_SLC_metrics(y_true, y_pred)

        return test_scores

    def train(self, args, round, model_weights):
        missing_keys, unexpected_keys = self.model.load_state_dict(model_weights, strict=False)
        print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))
        self.model = self.model.to(self.device)

        self.local_update(args, round)

        updated_weights = {k: p.cpu() for k, p in self.model.named_parameters()}
        # client testing
        test_scores = self.evaluate(args, self.model)

        return updated_weights, test_scores