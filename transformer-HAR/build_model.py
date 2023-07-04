from network import TransformerEncoder, LSTMEncoder, Classifier


def build_model(d_model, data_feature_size, n_class, nhead, num_encoder_layers, dim_feedforward, dropout, encoder='transformer', do_input_embedding=False):
    if encoder == 'transformer':
        data_encoder = TransformerEncoder(
            d_model=d_model,
            in_vocab_size=data_feature_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            do_input_embedding=do_input_embedding
        )
    elif encoder == 'lstm':
        data_encoder = LSTMEncoder(
            data_feature_size,
            emb_size=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout,
            do_input_embedding=do_input_embedding
        )
    else:
        raise ValueError('Wrong model name')

    model = Classifier(data_encoder, emb_dim=d_model, out_dim=n_class)

    return model