from collections import defaultdict
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, in_vocab_size, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, do_input_embedding=False):
        super(TransformerEncoder, self).__init__()
        self.nhead = nhead
        if d_model % self.nhead != 0:
            d_model = d_model + self.nhead - d_model % self.nhead

        self.d_model = d_model
        self.do_input_embedding = do_input_embedding # indicate whether the inputs are ID and need to do embedding
        if do_input_embedding:
            self.enc_embedding = nn.Embedding(in_vocab_size, d_model)
        else:
            self.enc_embedding = nn.Linear(in_vocab_size, d_model)
        self.pos_embedding_enc = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, norm=encoder_norm)

    def make_src_mask(self, inp):
        if self.do_input_embedding:
            return inp.transpose(0, 1) == 0
        else:
            return torch.all(inp == 0, dim=-1).transpose(0, 1)


    def forward(self, src, output_type='avg'):
        # src: [src_len, batch_size, feature_dim]
        src_pad_mask = self.make_src_mask(src)

        src = self.enc_embedding(src)
        src = self.pos_embedding_enc(src)  # [src_len, batch_size, embed_dim]
        memory = self.encoder(src=src, mask=None, src_key_padding_mask=src_pad_mask) # padding marker

        seq_len = (~src_pad_mask).sum(-1)
        memory = torch.mul(memory, ~src_pad_mask.repeat(self.d_model, 1, 1).permute(2, 1, 0))

        # [src_len, batch_size, embed_dim]
        if output_type == 'sum':
            embedding = torch.sum(memory, dim=0)
        elif output_type == 'avg':
            embedding = torch.sum(memory, dim=0) / seq_len.unsqueeze(-1)
        elif output_type == 'last':
            embedding = memory[[(seq_len-1).to(torch.long), torch.range(0, memory.size(1)-1).to(torch.long)]]  # the last timestep
        else:
            raise ValueError('Wrong value of output_type.')


        return embedding  # [batch_size, emb_dim]


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, emb_size=256, num_layers=1, dropout=0.5, do_input_embedding=False):
        super(BiLSTMEncoder, self).__init__()

        self.do_input_embedding = do_input_embedding # indicate whether the inputs are ID and need to do embedding
        if do_input_embedding:
            self.enc_embedding = nn.Embedding(input_size, emb_size)
            input_size = emb_size

        self.hidden_size = emb_size // 2
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.drop = nn.Dropout(p=dropout)

    def make_src_mask(self, inp):
        if self.do_input_embedding:
            return inp.transpose(0, 1) == 0
        else:
            return torch.all(inp == 0, dim=-1).transpose(0, 1)

    def forward(self, src, output_type='avg'):
        src_pad_mask = self.make_src_mask(src)
        src_len = (~src_pad_mask).sum(-1)
        if self.do_input_embedding:
            src = self.enc_embedding(src)
        packed_input = pack_padded_sequence(src, src_len.cpu().numpy(), batch_first=False, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=False) # [seq_len, batch_size, hidden_size]

        if output_type == 'last':
            lstm_out_forward = output[src_len - 1, range(len(src_len)), :self.hidden_size]
            lstm_out_reverse = output[0, :, self.hidden_size:]
        elif output_type == 'avg':
            lstm_out_forward = torch.sum(output[:, :, :self.hidden_size], dim=0) / src_len.unsqueeze(-1)
            lstm_out_reverse = torch.sum(output[:, :, self.hidden_size:], dim=0) / src_len.unsqueeze(-1)
        else:
            raise ValueError('Wrong value of output_type.')

        output = torch.cat((lstm_out_forward, lstm_out_reverse), 1)

        return output

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, emb_size=256, num_layers=1, dropout=0.5, do_input_embedding=False):
        super(LSTMEncoder, self).__init__()

        self.do_input_embedding = do_input_embedding # indicate whether the inputs are ID and need to do embedding
        if do_input_embedding:
            self.enc_embedding = nn.Embedding(input_size, emb_size)
            input_size = emb_size

        self.hidden_size = emb_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=emb_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=False,
        )
        self.drop = nn.Dropout(p=dropout)

    def make_src_mask(self, inp):
        if self.do_input_embedding:
            return inp.transpose(0, 1) == 0
        else:
            return torch.all(inp == 0, dim=-1).transpose(0, 1)

    def forward(self, src, output_type='avg'):
        #print(src.shape)
        src_pad_mask = self.make_src_mask(src)
        src_len = (~src_pad_mask).sum(-1)
        if self.do_input_embedding:
            src = self.enc_embedding(src)
        packed_input = pack_padded_sequence(src, src_len.cpu().numpy(), batch_first=False, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=False) # [seq_len, batch_size, hidden_size]

        if output_type == 'last':
            output = output[src_len - 1, range(len(src_len)), :self.hidden_size]
        elif output_type == 'avg':
            output = torch.sum(output[:, :, :self.hidden_size], dim=0) / src_len.unsqueeze(-1)
        else:
            raise ValueError('Wrong value of output_type.')

        return output


class ConventionalClassifier(nn.Module):
    def __init__(self, data_encoder, emb_dim, out_dim): #, init_last_layer=None):
        super(ConventionalClassifier, self).__init__()
        self.data_encoder = data_encoder
        self.classifier = nn.Linear(emb_dim, out_dim)

    def forward(self, x_data):
        # x_data: [src_len, batch_size, feature_dim]
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        out = self.classifier(z_data)
        return out

class LocalClassifier(nn.Module):
    def __init__(self, n_feature, n_output, nonlinearity=False):
        super().__init__()
        self.nonlinearity = nonlinearity
        layers = []
        if nonlinearity:
            layers.append(nn.ReLU())

        layers.append(nn.Linear(n_feature, n_output))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, last_dim=256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, last_dim)
        self.fc2 = nn.Linear(last_dim, last_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ConvNet(nn.Module):
    def __init__(self, in_channels, n_kernels, emb_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.classifier(x)

        return out


class HyperNet(nn.Module):
    def __init__(self, n_client, target_network_weights, hidden_dim, n_hidden, spec_norm=False):
        super().__init__()

        # client embedding
        emb_layers = [
            nn.Embedding(num_embeddings=n_client, embedding_dim=hidden_dim),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            emb_layers.append(nn.ReLU(inplace=True))
            emb_layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        self.embeddings = nn.Sequential(*emb_layers)

        # generate target network weights
        self.module2weight = defaultdict(str)  # name of module in hypernet -> name in state_dict
        self.layer_sizes = {}

        for k in target_network_weights:
            if len(target_network_weights[k].size()):
                if spec_norm:
                    layer = spectral_norm(nn.Linear(hidden_dim, np.prod([[s for s in target_network_weights[k].size()]])))
                else:
                    layer = nn.Linear(hidden_dim, np.prod([[s for s in target_network_weights[k].size()]]))

            module_name = k.replace('.', '@')
            self.module2weight[module_name] = k
            self.add_module(module_name, layer)
            self.layer_sizes[k] = target_network_weights[k].size()

    def forward(self, indices):
        features = self.embeddings(indices)

        weights = {}
        for module_name, layer in self.named_modules():
            if self.module2weight[module_name] in self.layer_sizes:
                weights[self.module2weight[module_name]] = layer(features).view(-1, *self.layer_sizes[self.module2weight[module_name]])

        outputs = []
        for i in range(len(indices)):
            outputs.append({k: weights[k][i] for k in weights})

        return outputs


def adjust_net(net, large_input=False):
    """
    Adjusts the first layers of the network so that small images (32x32) can be processed.
    :param net: neural network
    :param large_input: True if the input images are large (224x224 or more).
    :return: the adjusted network
    """
    net.expected_input_sz = 224 if large_input else 32

    if large_input:
        return net

    def adjust_first_conv(conv1, ks=(3, 3), stride=1):
        assert conv1.in_channels == 3, conv1
        ks_org = conv1.weight.data.shape[2:]
        if ks_org[0] > ks[0] or ks_org[1] or ks[1]:
            # use the center of the filters
            offset = ((ks_org[0] - ks[0]) // 2, (ks_org[1] - ks[1]) // 2)
            offset1 = ((ks_org[0] - ks[0]) % 2, (ks_org[1] - ks[1]) % 2)
            conv1.weight.data = conv1.weight.data[:, :, offset[0]:-offset[0]-offset1[0], offset[1]:-offset[1]-offset1[1]]
            assert conv1.weight.data.shape[2:] == ks, (conv1.weight.data.shape, ks)
        conv1.kernel_size = ks
        conv1.padding = (ks[0] // 2, ks[1] // 2)
        conv1.stride = (stride, stride)

    if isinstance(net, ResNet):

        adjust_first_conv(net.conv1)
        assert hasattr(net, 'maxpool'), type(net)
        net.maxpool = nn.Identity()

    elif isinstance(net, DenseNet):

        adjust_first_conv(net.features[0])
        assert isinstance(net.features[3], nn.MaxPool2d), (net.features[3], type(net))
        net.features[3] = nn.Identity()

    elif isinstance(net, (MobileNetV2, MobileNetV3)):  # requires torchvision 0.9+

        def reduce_stride(m):
            if isinstance(m, nn.Conv2d):
                m.stride = 1

        for m in net.features[:5]:
            m.apply(reduce_stride)

    elif isinstance(net, VGG):

        for layer, mod in enumerate(net.features[:10]):
            if isinstance(mod, nn.MaxPool2d):
                net.features[layer] = nn.Identity()

    elif isinstance(net, AlexNet):

        net.features[0].stride = 1
        net.features[2] = nn.Identity()

    elif isinstance(net, MNASNet):

        net.layers[0].stride = 1

    elif isinstance(net, ShuffleNetV2):

        net.conv1.stride = 1
        net.maxpool = nn.Identity()

    elif isinstance(net, GoogLeNet):

        net.conv1.stride = 1
        net.maxpool1 = nn.Identity()

    else:
        print('WARNING: the network (%s) is not adapted for small inputs which may result in lower performance' % str(
            type(net)))

    return net