from sqlalchemy import false
import torch
import numpy as np
import dataset
import config
from tqdm import tqdm
from resnet import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.cuda.amp.GradScaler()


# %% ---------------------------------------------
train_dataset = dataset.CatDogDataset(config.train_foder, transforms=dataset.train_aug())
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)


# %% ---------------------------------------------
net = ResNet18()
net.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


# %% ---------------------------------------------
def train_step(batch):
    optimizer.zero_grad()
    img, label = batch[0].to(device), batch[1].to(device)
    with torch.cuda.amp.autocast():
        out = net(img)
    loss = loss_fn(out, label)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    _preds = torch.argmax(out, axis=1)
    _preds = (_preds == label) * 1.
    acc = torch.mean(_preds)
    return loss, acc


# %% ---------------------------------------------
EPOCHS = 50
log = []
steps_per_epoch = len(train_dataloader)
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}/{EPOCHS}')
    net.train()
    train_loss = []
    train_acc = []
    t_loader = tqdm(train_dataloader)
    for train_data in t_loader:
        loss, acc = train_step(train_data)
        train_loss.append(loss.item())
        train_acc.append(acc.item())

    print(f'train loss: {np.mean(train_loss):.3f} - train acc: {np.mean(train_acc):.3f}')
    log.append(
        {
            'step': (epoch + 1) * steps_per_epoch,
            'loss': np.mean(train_loss),
            'acc': np.mean(train_acc)
        }
    )

import pandas as pd
df = pd.DataFrame(log)
df.to_csv('resnet.csv', index=False)