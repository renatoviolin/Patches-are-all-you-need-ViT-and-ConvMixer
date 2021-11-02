# %% ---------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %% ---------------------------------------------
conv1 = pd.read_csv('conv_1.csv')
conv2 = pd.read_csv('conv_2.csv')
conv3 = pd.read_csv('conv_3.csv')
vit1 = pd.read_csv('vit_1.csv')
vit2 = pd.read_csv('vit_2.csv')
vit3 = pd.read_csv('vit_3.csv')
vit4 = pd.read_csv('vit_4.csv')
resnet = pd.read_csv('resnet.csv')

resnet = np.array(resnet['loss'])
vit1 = np.array(vit1['loss'])
vit2 = np.array(vit2['loss'])
vit3 = np.array(vit3['loss'])
vit4 = np.array(vit4['loss'])
conv1 = np.array(conv1['loss'])
conv2 = np.array(conv2['loss'])
conv3 = np.array(conv3['loss'])

x = np.arange(50)

plt.title('ViT x ConvMixer')
plt.plot(x, resnet, '--', label=f'Resnet18')
plt.plot(x, vit1, label=f'Vit1: p24/Linear')
plt.plot(x, vit2, label=f'Vit2: p24/CNN')
plt.plot(x, vit3, label=f'Vit3: p16/Linear')
plt.plot(x, vit4, label=f'Vit4: p16/Linear')
plt.plot(x, conv1, ':', label=f'Conv1: p24')
plt.plot(x, conv2, ':', label=f'Conv2: p16')
plt.plot(x, conv3, ':', label=f'Conv3: p16/8')

plt.xlabel('epoch')
plt.ylabel('train loss')
plt.legend()
plt.savefig('loss.jpg', dpi=300)

# %%
