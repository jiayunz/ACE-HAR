import os
import torch
from torch import nn
import numpy as np
from sklearn.metrics import classification_report
from build_model import build_model
from load_data import load_data, collate_fn
from torch.utils.data import DataLoader

EPS = 1e-7
DATA_DIR = 'data'
SEED = 4321

def evaluate(model, data_loader, device):
    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for sample, label in data_loader:
            sample = sample.to(device).transpose(0, 1)
            label = label.to(device, dtype=torch.float)
            out = model(sample)
            out = torch.softmax(out, dim=-1)

            y_pred.extend(out.cpu().numpy())
            y_true.extend(label.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred


def train_one_batch(model, sample, label, optimizer, device):
    criterion = nn.CrossEntropyLoss()

    sample = sample.to(device).transpose(0, 1)
    label = label.to(device, dtype=torch.float)
    optimizer.zero_grad()
    out = model(sample)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    trainData, valData, testData = load_data(os.path.join(DATA_DIR, 'PAMAP2/pamap2_data_100.pkl'), os.path.join(DATA_DIR, 'PAMAP2/pamap2_label_100.pkl'), seed=SEED)
    train_loader = DataLoader(testData, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=1)
    test_loader = DataLoader(testData, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=1)

    model = build_model(
        d_model=64,
        data_feature_size=np.shape(testData.data[0])[-1],
        n_class=12,
        nhead=4,
        num_encoder_layers=1,
        dim_feedforward=64,
        dropout=0.5,
        encoder='transformer',
        do_input_embedding=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.005 for mimic3
    model.train()
    local_loss = []
    for e in range(50):
        for sample, label in train_loader:
            avg_sample_loss = train_one_batch(model, sample, label, optimizer, device)
            local_loss.append(avg_sample_loss)

        y_true, y_pred = evaluate(model, test_loader, device)
        print(classification_report(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()