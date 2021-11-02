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
efficient = pd.read_csv('efficient.csv')

efficient_loss = np.array(efficient['loss'])
resnet_loss = np.array(resnet['loss'])
vit1_loss = np.array(vit1['loss'])
vit2_loss = np.array(vit2['loss'])
vit3_loss = np.array(vit3['loss'])
vit4_loss = np.array(vit4['loss'])
conv1_loss = np.array(conv1['loss'])
conv2_loss = np.array(conv2['loss'])
conv3_loss = np.array(conv3['loss'])


x = np.arange(50)
plt.title('ViT x ConvMixer')
plt.plot(x, efficient_loss, '--', label=f'EfficientNetV2-B0')
plt.plot(x, resnet_loss, '--', label=f'Resnet18')
plt.plot(x, vit1_loss, label=f'Vit1: p24/Linear')
plt.plot(x, vit2_loss, label=f'Vit2: p24/CNN')
plt.plot(x, vit3_loss, label=f'Vit3: p16/Linear')
plt.plot(x, vit4_loss, label=f'Vit4: p16/CNN')
plt.plot(x, conv1_loss, label=f'Conv1: p24')
plt.plot(x, conv2_loss, label=f'Conv2: p16')
plt.plot(x, conv3_loss, label=f'Conv3: p16/8')

plt.xlabel('epoch')
plt.ylabel('train loss')
plt.rc('legend', fontsize=8)
plt.legend()
plt.savefig('loss.jpg', dpi=300)
plt.show()

# %% ---------------------------------------------
efficient_acc = np.array(efficient['acc'])
resnet_acc = np.array(resnet['acc'])
vit1_acc = np.array(vit1['acc'])
vit2_acc = np.array(vit2['acc'])
vit3_acc = np.array(vit3['acc'])
vit4_acc = np.array(vit4['acc'])
conv1_acc = np.array(conv1['acc'])
conv2_acc = np.array(conv2['acc'])
conv3_acc = np.array(conv3['acc'])


x = np.arange(50)
plt.title('ViT x ConvMixer')
plt.plot(x, efficient_acc, '--', label=f'EfficientNetV2-B0')
plt.plot(x, resnet_acc, '--', label=f'Resnet18')
plt.plot(x, vit1_acc, label=f'Vit1: p24/Linear')
plt.plot(x, vit2_acc, label=f'Vit2: p24/CNN')
plt.plot(x, vit3_acc, label=f'Vit3: p16/Linear')
plt.plot(x, vit4_acc, label=f'Vit4: p16/CNN')
plt.plot(x, conv1_acc, label=f'Conv1: p24')
plt.plot(x, conv2_acc, label=f'Conv2: p16')
plt.plot(x, conv3_acc, label=f'Conv3: p16/8')

plt.xlabel('epoch')
plt.ylabel('train acc')
plt.rc('legend', fontsize=8)
plt.legend()
plt.savefig('acc.jpg', dpi=300)
plt.show()
