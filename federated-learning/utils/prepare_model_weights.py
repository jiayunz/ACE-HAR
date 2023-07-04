
def convert_model_key_to_idx(global_key_to_idx, model_name, model_weights):
    # todo: personalized classifier
    converted_model_weights = {}
    for k, p in model_weights.items():
        if k in global_key_to_idx:
            global_k_idx = global_key_to_idx[k]
        elif model_name + '.' + k in global_key_to_idx:
            global_k_idx = global_key_to_idx[model_name + '.' + k]
        else:
            #print(f'Not found {k} in global_key_to_idx')
            continue

        converted_model_weights[global_k_idx] = p.cpu()

    return converted_model_weights


def prepare_client_weights(model, model_name, recv_weight):
    new_model_weights = {}
    for k, p in model.named_parameters():
        if not p.requires_grad: # todo: check
            continue
        elif model_name + '.' + k in recv_weight:
            global_k = model_name + '.' + k
            if p.size() == recv_weight[global_k].size():
                new_model_weights[k] = recv_weight[global_k].cpu()
        elif k in recv_weight and p.size() == recv_weight[k].size():
            new_model_weights[k] = recv_weight[k].cpu()
        #else:
        #    print(f'Not found {k} in recv_weight')

    return new_model_weights