# CV-lab-problem-A (Line fitting problem)(2020/01/26)
## Fetch data (unchanged)
```
!wget -nc 140.114.85.52:8000/pA1.csv
!wget -nc 140.114.85.52:8000/pA2.csv
```
## Preparations (unchanged)
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D #for 3D visualization

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

seed = 999
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```
## Dataset and Training
- Briefings
  - Add dataloader for the second dataset
  - Add second model for the second dataset
  - Plot for the 3D visualization of model 1 performance (x_axis=a, y_axis=b, z_axis=loss)
  - Plot for the 2D visualization of model 2 performance (x_axis=epoch_index, y_axis=loss)
  - Adjust learning rate and epoch numbers
```
class Data:
    def __init__(self, csv_path):
        super().__init__()
        self.anns = pd.read_csv(csv_path).to_dict('records')

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        x = torch.tensor(ann['x'])
        y = torch.tensor(ann['y'])
        return x, y


class Net1(nn.Module): # model 1 with forward y=ax+b
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1) * 0.001)
        self.b = nn.Parameter(torch.rand(1) * 0.001)
    
    def forward(self, xs):
        ps = self.a * xs + self.b
        return ps

class Net2(nn.Module):  # model 2 with forward nn.linear() y=ax^2+bx+c
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.rand(1) * 0.001)
        self.b = nn.Parameter(torch.rand(1) * 0.001)
        self.c = nn.Parameter(torch.rand(1) * 0.001)
    
    def forward(self, xs):
        ps = self.a * xs * xs + self.b *xs + self.c
        return ps        


loader1 = DataLoader(Data('./pA1.csv'), batch_size=1, shuffle=True)
loader2 = DataLoader(Data('./pA2.csv'), batch_size=1, shuffle=True) #datalader for model 2

device = 'cpu'
model1 = Net1().to(device)


criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-1)


history1 = {
    'loss': [],
    'a': [],
    'b': []
}

history2 = {
    'loss': [],
    'a': [],
    'b': [],
    'c': [],
    'idx':[]
}

# Record a, b, loss for 3D visualization of model 1
plot1 = {
    'loss': [],
    'a':[],
    'b':[]
}

# Record loss and epoch index for visualization of model2
plot2 = {
    'loss': [],
    'idx':[]
}

for epoch in range(20):
    totalloss = 0
    for xs, ys in iter(loader1):
        xs = xs.to(device)
        ys = ys.to(device)

        optimizer.zero_grad()
        ps = model1(xs)
        loss1 = criterion(ps, ys)
        loss1.backward()
        optimizer.step()

        totalloss = totalloss + loss1.detach().item()

        history1['loss'].append(loss1.detach().item())
        history1['a'].append(model1.a.item())
        history1['b'].append(model1.b.item())
    plot1['loss'].append(totalloss/len(loader1))
    plot1['a'].append(model1.a.item())
    plot1['b'].append(model1.b.item())


print(model1.a)
print(model1.b)

# Plot of model 1 performance
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(plot1['a'], plot1['b'], plot1['loss'], label='loss curve to parameters')
ax.legend()
ax.set_xlabel('a')
ax.set_ylabel('b') 
ax.set_zlabel('L')

plt.show()

model2 = Net2().to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-1)
for epoch in range(20):
    totalloss = 0
    for xs2, ys2 in iter(loader2):
        xs2 = xs2.to(device)
        ys2 = ys2.to(device)

        optimizer.zero_grad()
        ps2 = model2(xs2)
        loss2 = criterion(ps2, ys2)
        loss2.backward()
        optimizer.step()

        totalloss = totalloss + loss2.detach().item()

        history2['loss'].append(loss2.detach().item())
        history2['a'].append(model2.a.item())
        history2['b'].append(model2.b.item())
        history2['c'].append(model2.c.item())
    plot2['idx'].append(epoch)
    plot2['loss'].append(totalloss/len(loader2))

    

print(model2.a)
print(model2.b)
print(model2.c)

print(plot2['idx'])
print(plot2['loss'])

# Plot of model 2 performance
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(plot2['idx'], plot2['loss'])
plt.show()
```
## Results
- Dataset 1 (y=ax+b)
  - a = 4.9948(5), b=4.1474(4)
```
Parameter containing:
tensor([4.9948], requires_grad=True)
Parameter containing:
tensor([4.1474], requires_grad=True)
```
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/loss_model1.png "loss_model1")
- Dataset 2 (y=ax^x+bx+c)
  - a = -1.8965(-2), b=1.2241(1.2), c=4.0224(4)
```
Parameter containing:
tensor([-1.8965], requires_grad=True)
Parameter containing:
tensor([1.2241], requires_grad=True)
Parameter containing:
tensor([4.0224], requires_grad=True)
```
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/loss_model2.png "loss_model2")

## Colab link
https://colab.research.google.com/drive/1K_uQCahGZVw10YwBH-182OslLaxC4oic#scrollTo=qwVCZ2ZCCwRb <br>
colab version outputed in github as well


# CV-lab-problem-B (Car plate recongization problem)
## Path.py
----------------
- 使用自己的方式取出圖片的 ground truth <br>
  逐一拆解 "_" "&" "-" (利用join() spilt()處理字串)
- 此為之後讀資料的先行嘗試 <br>
- 註解者為範例code與notes
```python
from pathlib import Path

img_dir = Path('./ccpd5000/train/')
img_paths_all = img_dir.glob('*.jpg')
img_paths_list = list(img_paths_all)
img_paths = sorted(img_paths_list) #sorted image paths
#img_paths = img_dir.glob('*.jpg')
#img_paths = sorted(list(img_paths))


print('data size: '+str(len(img_paths)))
print('sample image name: ' + str(img_paths[8]))
#print(len(img_paths))
#print(img_paths[:5])

testname = img_paths[8].name
print(testname)

split_component = testname.split('-')
information = split_component[3]
## = name.split('-')[3]
##print(token)

information_split1 = "&".join(information.split('_'))
information_split2 = information_split1.split('&')
#token = token.replace('&', '_')
#print(token)

#values = token.split('_')
#print(values)

values = list()
for i in range(len(information_split2)):
    values.append(float(information_split2[i]))
print(values)

#values = [float(val) for val in values]
#print(values)

```
## util.py
- 繪圖與轉換的程式碼 <br> (保持與範例原樣)
```python
# UNCHANGED

import warnings

import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import util
from skimage.transform import ProjectiveTransform, warp

def draw_kpts(img, kpts, c='red', r=2.0):
    '''Draw keypoints on image.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
        c: (PIL.Color) color of keypoints, default to 'red'
        r: (float) radius of keypoints, default to 2.0
    Return:
        img: (PIL.Image) modified image
    '''
    draw = ImageDraw.Draw(img)
    kpts = kpts.view(4, 2)
    kpts = kpts * torch.FloatTensor(img.size)
    kpts = kpts.numpy().tolist()
    for (x, y) in kpts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


def draw_plate(img, kpts):
    '''Perspective tranform and draw the plate indicated by kpts to a 96x30 rectangle.
    Args:
        img: (PIL.Image) will be modified
        kpts: (FloatTensor) keypoints in xy format, sized [8,]
    Return:
        img: (PIL.Image) modified image
    Reference: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_geometric.html
    '''
    src = np.float32([[96, 30], [0, 30], [0, 0], [96, 0]])
    dst = kpts.view(4, 2).numpy()
    dst = dst * np.float32(img.size)

    transform = ProjectiveTransform()
    transform.estimate(src, dst)
    with warnings.catch_warnings(): # surpress skimage warning
        warnings.simplefilter("ignore")
        warped = warp(np.array(img), transform, output_shape=(30, 96))
        warped = util.img_as_ubyte(warped)
    plate = Image.fromarray(warped)
    img.paste(plate)
    return img
```

## data.py
- 與 path.py 之延伸，用以讀出data <br>
- 覆寫 __init__() __len__() __getitem__()函數
  - 以 for 迴圈將 grond truth 的長寬 normalize
  - 讀出 image 那段似乎是特定函數用法，保持不動

```python
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf



class CCPD5000:
    def __init__(self, img_dir):  # use CCPD5000(path) initiate
        self.img_dir = Path(img_dir)

        self.img_paths_all = self.img_dir.glob("*.jpg")
        self.img_paths_list = list(self.img_paths_all)
        self.img_paths = sorted(self.img_paths_list)  # sorted image path

    #    def __init__(self, img_dir):
    #        self.img_dir = Path(img_dir)
    #        self.img_paths = self.img_dir.glob('*.jpg')
    #        self.img_paths = sorted(list(self.img_paths))

    def __len__(self):  # return size
        return len(self.img_paths)

#    def __len__(self):
#        return len(self.img_paths)

    def __getitem__(self, index):  # call with CCPD5000((int)index) return [img , [8, ](tensor)ground_truth]
        target_img_path = self.img_paths[index]
        target_img = Image.open(target_img_path)
        (W, H) = target_img.size

        # get [8, ] ground truth
        img_name = target_img_path.name
        spilt_component = img_name.split("-")
        information = spilt_component[3]
        information_split1 = "&".join(information.split("_"))
        information_split2 = information_split1.split("&")

        ground_truth = list()
        for i in range(len(information_split2)):
            ground_truth.append(float(information_split2[i]))

        for idx in range(len(ground_truth)):
            if(idx%2==0):
                ground_truth[idx] = ground_truth[idx]/W
            else:
                ground_truth[idx] = ground_truth[idx]/H

        ground_truth = torch.tensor(ground_truth)

        # adjust image UNCHANGED
        target_img = target_img.convert("RGB")
        target_img = target_img.resize((192, 320))
        target_img = tf.to_tensor(target_img)

        return (target_img, ground_truth)



#    def __getitem__(self, idx):
#        img_path = self.img_paths[idx]

# load image
#        img = Image.open(img_path)
#        W, H = img.size
#        img = img.convert('RGB')
#        img = img.resize((192, 320))
#        img = tf.to_tensor(img)

# parse annotation
#        name = img_path.name
#        token = name.split('-')[3]
#        token = token.replace('&', '_')
#        kpt = [float(val) for val in token.split('_')]
#        kpt = torch.tensor(kpt)  # [8,]
#        kpt = kpt.view(4, 2)  # [4, 2]
#        kpt = kpt / torch.FloatTensor([W, H])
#        kpt = kpt.view(-1)  # [8,]

#        return img, kpt


train_set = CCPD5000('./ccpd5000/train')
print(len(train_set))

img, kpt = train_set[90]
print(img.size())
print(kpt.size())

print(kpt)

# image display for colab (disable for pycharm)

#img = tf.to_pil_image(img)
#vis = draw_kpts(img, kpt, c='orange')
#vis = draw_plate(vis, kpt)
#vis.save('./check.jpg')

#from IPython import display
#display.Image('./check.jpg')
```

## convolution_block.py
- 由於對 nn.Sequential() 用法不甚熟悉，所以用比較熟的 net() 用法慢慢架
- 重新設計 forward() 方法
  - convolution2D ->batch_norm2d -> leakyRelu -> maxPooling2d 為一週期
    - 4 cycles
    - 每次 maxpooling 的長寬更小(比起原先8*8)
  - 擬合函數
    - 三層 linear function 讓 tensor size reduce from 128 -> 8
```python
import torch
from torch import nn
from torch.nn import functional as F

class CCPDRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(4, 8, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(8, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.batchN2_1 = nn.BatchNorm2d(4)
        self.batchN2_2 = nn.BatchNorm2d(8)
        self.batchN2_3 = nn.BatchNorm2d(16)
        self.batchN2_4 = nn.BatchNorm2d(32)
        self.batchN2_5 = nn.BatchNorm2d(64)
        self.batchN2_6 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.leakyR = nn.LeakyReLU()
        self.sigmo = nn.Sigmoid()

    def forward(self, x):
        N = x.size(0)

        x = self.leakyR(self.batchN2_1(self.conv1(x)))
        x = F.max_pool2d(x, (4, 4))
        x = self.leakyR(self.batchN2_2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_3(self.conv3(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_4(self.conv4(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_5(self.conv5(x)))
        x = F.max_pool2d(x, (2, 2))
        x = self.leakyR(self.batchN2_6(self.conv6(x)))
        x = F.max_pool2d(x, (2, 2))

        #print(x.size())
        x = x.view(N, -1)

        x = self.leakyR(self.fc1(x))
        x = self.leakyR(self.fc2(x))
        x = self.sigmo(self.fc3(x))

        #print(x.size())
        return x



'''class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(cout, cout, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act1(self.bn2(self.conv2(x)))
        return x


class CCPDRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            nn.MaxPool2d((4, 4)),
            ConvBlock(16, 32),
            nn.MaxPool2d((4, 4)),
            ConvBlock(32, 64),
            nn.MaxPool2d(4, 4),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = x.view(N, -1)  # i.e. Flatten
        x = self.regressor(x)
        return x

'''
# Check UNCHANGED
device = 'cuda'
model = CCPDRegressor().to(device)
img_b = torch.rand(16, 3, 192, 320).to(device)
out_b = model(img_b)
print(out_b.size())  # expected [16, 8]
```
## train.py
  - 似乎許多都是固定架構與函式用法，重新寫了一次當練習(當作熟悉架構(?)
  - loss 改成以 MSE 評估
  - learning rate 改成2e-4(原先的兩倍(不然實測有些慢))
  - epoch 數改為(30)
    - train 與 vaild 的誤差(mae,mse)皆有持續下降趨勢(epoch = 22 後趨於不明顯)
    - 最低值恰好在 epoch = 29 時
      - TRAIN: avg_mae=0.00791, avg_mse=0.00011
      - VAILD: avg_mae=0.01147, avg_mse=0.00029
  - 顯示進度條與顯示圖片的code皆不動
```python
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import functional as tf

# For reproducibility
# Set before loading model and dataset UNCHANGED
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Load data UNCHANGED
train_set = CCPD5000('./ccpd5000/train/')
valid_set = CCPD5000('./ccpd5000/valid/')
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 32)),
    Subset(valid_set, random.sample(range(len(valid_set)), 32)),
])
train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=3)
valid_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=1)
visul_loader = DataLoader(visul_set, 32, shuffle=False, num_workers=1)

device = 'cuda'
model = CCPDRegressor()
model = model.to(device)
criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
#device = 'cuda'
#model = CCPDRegressor().to(device)
#criterion = nn.MSELoss().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Log record UNCHANGED
log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)
print(log_dir)
history = {
    'train_mae': [],
    'valid_mae': [],
    'train_mse': [],
    'valid_mse': [],
}


# train
def train(pbar):
    model.train()  # train mode
    mae_steps = []
    mse_steps = []

    for image, ground_truth in iter(train_loader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)

        optimizer.zero_grad()
        predict = model(image)
        loss = criterion(predict, ground_truth)
        loss.backward()
        optimizer.step()

        mae = F.l1_loss(predict, ground_truth).item()
        mse = F.mse_loss(predict, ground_truth).item()

        # UNCHANGED
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(image.size(0))
    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['train_mae'].append(avg_mae)
    history['train_mse'].append(avg_mse)

'''def train(pbar):
    model.train()
    mae_steps = []
    mse_steps = []

    for img_b, kpt_b in iter(train_loader):
        img_b = img_b.to(device)
        kpt_b = kpt_b.to(device)

        optimizer.zero_grad()
        pred_b = model(img_b)
        loss = criterion(pred_b, kpt_b)
        loss.backward()
        optimizer.step()

        mae = loss.detach().item()
        mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(img_b.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['train_mae'].append(avg_mae)
    history['train_mse'].append(avg_mse)
'''

def valid(pbar):
    model.eval()  # evaluation mode
    mae_steps = []
    mse_steps = []

    for image, ground_truth in iter(valid_loader):
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        predict = model(image)
        loss = criterion(predict, ground_truth)

        mae = F.l1_loss(predict, ground_truth).item()
        mse = F.mse_loss(predict, ground_truth).item()
        # UNCHANGED
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(image.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['valid_mae'].append(avg_mae)
    history['valid_mse'].append(avg_mse)
'''
def valid(pbar):
    model.eval()
    mae_steps = []
    mse_steps = []

    for img_b, kpt_b in iter(valid_loader):
        img_b = img_b.to(device)
        kpt_b = kpt_b.to(device)
        pred_b = model(img_b)
        loss = criterion(pred_b, kpt_b)
        mae = loss.detach().item()

        mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
        mae_steps.append(mae)
        mse_steps.append(mse)

        pbar.set_postfix(mae=mae, mse=mse)
        pbar.update(img_b.size(0))

    avg_mae = sum(mae_steps) / len(mae_steps)
    avg_mse = sum(mse_steps) / len(mse_steps)
    pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
    history['valid_mae'].append(avg_mae)
    history['valid_mse'].append(avg_mse)
'''

# Visualization UNCHANGED
def visul(pbar, epoch):
    model.eval()
    epoch_dir = log_dir / f'{epoch:03d}'
    epoch_dir.mkdir()
    for img_b, kpt_b in iter(visul_loader):
        pred_b = model(img_b.to(device)).cpu()
        for img, pred_kpt, true_kpt in zip(img_b, pred_b, kpt_b):
            img = tf.to_pil_image(img)
            vis = draw_plate(img, pred_kpt)
            vis = draw_kpts(vis, true_kpt, c='orange')
            vis = draw_kpts(vis, pred_kpt, c='red')
            vis.save(epoch_dir / f'{pbar.n:03d}.jpg')
            pbar.update()

# log record UNCHANGED
def log(epoch):
    with (log_dir / 'metrics.json').open('w') as f:
        json.dump(history, f)

    fig, ax = plt.subplots(2, 1, figsize=(6, 6), dpi=100)
    ax[0].set_title('MAE')
    ax[0].plot(range(epoch + 1), history['train_mae'], label='Train')
    ax[0].plot(range(epoch + 1), history['valid_mae'], label='Valid')
    ax[0].legend()
    ax[1].set_title('MSE')
    ax[1].plot(range(epoch + 1), history['train_mse'], label='Train')
    ax[1].plot(range(epoch + 1), history['valid_mse'], label='Valid')
    ax[1].legend()
    fig.savefig(str(log_dir / 'metrics.jpg'))
    plt.close()


# train epoch setting UNCHANGED
for epoch in range(30):
    print('Epoch', epoch, flush=True)
    with tqdm(total=len(train_set), desc='  Train') as pbar:
        train(pbar)

    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid') as pbar:
            valid(pbar)
        with tqdm(total=len(visul_set), desc='  Visul') as pbar:
            visul(pbar, epoch)
        log(epoch)
```

## RESULT

![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/loss_graph.jpg "loss_graph")
```
log/2019.06.30-10:29:09
Epoch 0
  Train: 100%|██████████| 4000/4000 [00:41<00:00, 97.50it/s, avg_mae=0.10402, avg_mse=0.01855]
  Valid: 100%|██████████| 1000/1000 [00:12<00:00, 82.89it/s, avg_mae=0.09400, avg_mse=0.01424]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.29it/s]
Epoch 1
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 101.13it/s, avg_mae=0.07179, avg_mse=0.00890]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.61it/s, avg_mae=0.05519, avg_mse=0.00578]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.97it/s]
Epoch 2
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 116.92it/s, avg_mae=0.04458, avg_mse=0.00383]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.61it/s, avg_mae=0.04064, avg_mse=0.00330]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.50it/s]
Epoch 3
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 108.43it/s, avg_mae=0.03147, avg_mse=0.00199]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.13it/s, avg_mae=0.02727, avg_mse=0.00164]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.83it/s]
Epoch 4
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 102.12it/s, avg_mae=0.02244, avg_mse=0.00106]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.11it/s, avg_mae=0.02216, avg_mse=0.00109]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.00it/s]
Epoch 5
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 102.75it/s, avg_mae=0.01900, avg_mse=0.00075]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.12it/s, avg_mae=0.01994, avg_mse=0.00088]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.21it/s]
Epoch 6
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 103.41it/s, avg_mae=0.01681, avg_mse=0.00058]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.88it/s, avg_mae=0.01808, avg_mse=0.00073]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.88it/s]
Epoch 7
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 101.49it/s, avg_mae=0.01540, avg_mse=0.00048]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.20it/s, avg_mae=0.01632, avg_mse=0.00060]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.99it/s]
Epoch 8
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 102.87it/s, avg_mae=0.01407, avg_mse=0.00040]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.35it/s, avg_mae=0.01579, avg_mse=0.00054]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.72it/s]
Epoch 9
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 102.56it/s, avg_mae=0.01341, avg_mse=0.00035]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.86it/s, avg_mae=0.01470, avg_mse=0.00047]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 40.34it/s]
Epoch 10
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 103.56it/s, avg_mae=0.01247, avg_mse=0.00030]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.98it/s, avg_mae=0.01404, avg_mse=0.00043]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.45it/s]
Epoch 11
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 104.59it/s, avg_mae=0.01190, avg_mse=0.00027]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.64it/s, avg_mae=0.01337, avg_mse=0.00039]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.68it/s]
Epoch 12
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 104.58it/s, avg_mae=0.01135, avg_mse=0.00025]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.34it/s, avg_mae=0.01428, avg_mse=0.00043]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.82it/s]
Epoch 13
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 101.38it/s, avg_mae=0.01107, avg_mse=0.00023]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.62it/s, avg_mae=0.01325, avg_mse=0.00037]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.60it/s]
Epoch 14
  Train: 100%|██████████| 4000/4000 [00:37<00:00, 105.30it/s, avg_mae=0.01053, avg_mse=0.00021]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.28it/s, avg_mae=0.01277, avg_mse=0.00034]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.04it/s]
Epoch 15
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 109.29it/s, avg_mae=0.01031, avg_mse=0.00020]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.61it/s, avg_mae=0.01239, avg_mse=0.00034]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.98it/s]
Epoch 16
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 122.59it/s, avg_mae=0.01021, avg_mse=0.00019]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.55it/s, avg_mae=0.01208, avg_mse=0.00032]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 43.90it/s]
Epoch 17
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 109.42it/s, avg_mae=0.00980, avg_mse=0.00018]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.13it/s, avg_mae=0.01198, avg_mse=0.00032]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.58it/s]
Epoch 18
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 150.73it/s, avg_mae=0.00940, avg_mse=0.00016]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.65it/s, avg_mae=0.01190, avg_mse=0.00031]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 48.27it/s]
Epoch 19
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 104.79it/s, avg_mae=0.00936, avg_mse=0.00016]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.26it/s, avg_mae=0.01162, avg_mse=0.00030]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.59it/s]
Epoch 20
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 103.27it/s, avg_mae=0.00902, avg_mse=0.00015]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.91it/s, avg_mae=0.01305, avg_mse=0.00035]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.92it/s]
Epoch 21
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 107.20it/s, avg_mae=0.00908, avg_mse=0.00015]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.07it/s, avg_mae=0.01183, avg_mse=0.00030]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.79it/s]
Epoch 22
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 115.28it/s, avg_mae=0.00870, avg_mse=0.00014]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.85it/s, avg_mae=0.01240, avg_mse=0.00033]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.52it/s]
Epoch 23
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 130.74it/s, avg_mae=0.00846, avg_mse=0.00013]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.09it/s, avg_mae=0.01189, avg_mse=0.00032]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.15it/s]
Epoch 24
  Train: 100%|██████████| 4000/4000 [00:39<00:00, 102.53it/s, avg_mae=0.00833, avg_mse=0.00012]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.25it/s, avg_mae=0.01168, avg_mse=0.00030]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.28it/s]
Epoch 25
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 104.72it/s, avg_mae=0.00806, avg_mse=0.00012]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.89it/s, avg_mae=0.01166, avg_mse=0.00030]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.09it/s]
Epoch 26
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 103.38it/s, avg_mae=0.00804, avg_mse=0.00012]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.92it/s, avg_mae=0.01169, avg_mse=0.00030]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.77it/s]
Epoch 27
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 120.67it/s, avg_mae=0.00826, avg_mse=0.00012]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.11it/s, avg_mae=0.01190, avg_mse=0.00031]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.94it/s]
Epoch 28
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 105.11it/s, avg_mae=0.00798, avg_mse=0.00011]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.51it/s, avg_mae=0.01247, avg_mse=0.00033]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.31it/s]
Epoch 29
  Train: 100%|██████████| 4000/4000 [00:38<00:00, 107.80it/s, avg_mae=0.00791, avg_mse=0.00011]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.88it/s, avg_mae=0.01147, avg_mse=0.00029]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.39it/s]
  ```
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/plate_1.jpg "Logo plate_1")
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/plate_2.jpg "Logo plate_2")

## colab version
  https://colab.research.google.com/drive/17vCawLHzAhG_isiDweFo44uzkwJV1Pdv#scrollTo=Id4_SmMo9EuZ <br>
  colab version outputed in github as well
## Update 2020/01/26
  - In colab version <br>
    - add data augmentation <br> 
      - 增加上下顛倒(rotate 180 degree)之原圖片給train <br>
      - batch size from 16 -> 32
      - train set length 4000 -> 8000
      - adjust dataset and train codes to fit
  - Results
  ![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/loss_graph_0126.jpg "loss_graph_0126")
  ```
  log/2020.01.26-05:27:33
Epoch 0
  Train: 100%|██████████| 8000/8000 [01:26<00:00, 92.08it/s, avg_mae=0.08262, avg_mse=0.01154] 
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.96it/s, avg_mae=0.06558, avg_mse=0.00738]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 38.87it/s]
Epoch 1
  Train: 100%|██████████| 8000/8000 [01:26<00:00, 98.57it/s, avg_mae=0.05242, avg_mse=0.00494]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 83.78it/s, avg_mae=0.04746, avg_mse=0.00416]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.28it/s]
Epoch 2
  Train: 100%|██████████| 8000/8000 [01:25<00:00, 116.04it/s, avg_mae=0.03919, avg_mse=0.00287]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 85.36it/s, avg_mae=0.03641, avg_mse=0.00253]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 43.53it/s]
Epoch 3
  Train: 100%|██████████| 8000/8000 [01:23<00:00, 113.80it/s, avg_mae=0.03142, avg_mse=0.00192]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.49it/s, avg_mae=0.03256, avg_mse=0.00219]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.52it/s]
Epoch 4
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 128.33it/s, avg_mae=0.02665, avg_mse=0.00139]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.87it/s, avg_mae=0.02810, avg_mse=0.00160]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.47it/s]
Epoch 5
  Train: 100%|██████████| 8000/8000 [01:23<00:00, 95.66it/s, avg_mae=0.02371, avg_mse=0.00109]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 84.47it/s, avg_mae=0.02657, avg_mse=0.00149]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.18it/s]
Epoch 6
  Train: 100%|██████████| 8000/8000 [01:23<00:00, 129.12it/s, avg_mae=0.02168, avg_mse=0.00091]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.29it/s, avg_mae=0.02388, avg_mse=0.00120]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.82it/s]
Epoch 7
  Train: 100%|██████████| 8000/8000 [01:23<00:00, 95.81it/s, avg_mae=0.02023, avg_mse=0.00079]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 86.49it/s, avg_mae=0.02287, avg_mse=0.00111]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 43.94it/s]
Epoch 8
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 108.92it/s, avg_mae=0.01892, avg_mse=0.00068]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.82it/s, avg_mae=0.02247, avg_mse=0.00106]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.19it/s]
Epoch 9
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 117.08it/s, avg_mae=0.01801, avg_mse=0.00061]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.35it/s, avg_mae=0.02093, avg_mse=0.00090]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.25it/s]
Epoch 10
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 111.31it/s, avg_mae=0.01705, avg_mse=0.00055]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.44it/s, avg_mae=0.02009, avg_mse=0.00085]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 40.68it/s]
Epoch 11
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 96.78it/s, avg_mae=0.01626, avg_mse=0.00049]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.83it/s, avg_mae=0.01989, avg_mse=0.00082]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 43.85it/s]
Epoch 12
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 97.42it/s, avg_mae=0.01566, avg_mse=0.00045]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.60it/s, avg_mae=0.01941, avg_mse=0.00081]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 43.63it/s]
Epoch 13
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 97.55it/s, avg_mae=0.01520, avg_mse=0.00043]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.64it/s, avg_mae=0.01869, avg_mse=0.00075]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.44it/s]
Epoch 14
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 97.30it/s, avg_mae=0.01459, avg_mse=0.00039]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.68it/s, avg_mae=0.01807, avg_mse=0.00070]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.18it/s]
Epoch 15
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 97.60it/s, avg_mae=0.01402, avg_mse=0.00036]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 88.84it/s, avg_mae=0.01811, avg_mse=0.00069]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.76it/s]
Epoch 16
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 96.47it/s, avg_mae=0.01366, avg_mse=0.00034] 
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.90it/s, avg_mae=0.01779, avg_mse=0.00069]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 40.49it/s]
Epoch 17
  Train: 100%|██████████| 8000/8000 [01:22<00:00, 106.57it/s, avg_mae=0.01333, avg_mse=0.00032]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.90it/s, avg_mae=0.01786, avg_mse=0.00069]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 40.77it/s]
Epoch 18
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 98.16it/s, avg_mae=0.01295, avg_mse=0.00030]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.76it/s, avg_mae=0.01719, avg_mse=0.00065]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.01it/s]
Epoch 19
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 98.01it/s, avg_mae=0.01269, avg_mse=0.00029]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.39it/s, avg_mae=0.01704, avg_mse=0.00063]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.16it/s]
Epoch 20
  Train: 100%|██████████| 8000/8000 [01:20<00:00, 111.38it/s, avg_mae=0.01222, avg_mse=0.00027]
  Valid: 100%|██████████| 1000/1000 [00:10<00:00, 91.01it/s, avg_mae=0.01698, avg_mse=0.00062]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 46.38it/s]
Epoch 21
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 105.40it/s, avg_mae=0.01196, avg_mse=0.00026]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 90.04it/s, avg_mae=0.01697, avg_mse=0.00063]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.55it/s]
Epoch 22
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 102.71it/s, avg_mae=0.01172, avg_mse=0.00024]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.10it/s, avg_mae=0.01726, avg_mse=0.00066]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.20it/s]
Epoch 23
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 97.67it/s, avg_mae=0.01152, avg_mse=0.00024]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.15it/s, avg_mae=0.01597, avg_mse=0.00055]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.39it/s]
Epoch 24
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 98.16it/s, avg_mae=0.01124, avg_mse=0.00022]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.31it/s, avg_mae=0.01617, avg_mse=0.00056]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.52it/s]
Epoch 25
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 98.65it/s, avg_mae=0.01105, avg_mse=0.00022]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.33it/s, avg_mae=0.01599, avg_mse=0.00056]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 44.46it/s]
Epoch 26
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 98.27it/s, avg_mae=0.01078, avg_mse=0.00021]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.89it/s, avg_mae=0.01655, avg_mse=0.00061]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 47.61it/s]
Epoch 27
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 120.17it/s, avg_mae=0.01055, avg_mse=0.00019]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 89.19it/s, avg_mae=0.01604, avg_mse=0.00058]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 42.84it/s]
Epoch 28
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 101.95it/s, avg_mae=0.01041, avg_mse=0.00019]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 90.21it/s, avg_mae=0.01570, avg_mse=0.00055]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.49it/s]
Epoch 29
  Train: 100%|██████████| 8000/8000 [01:21<00:00, 99.39it/s, avg_mae=0.01017, avg_mse=0.00018]
  Valid: 100%|██████████| 1000/1000 [00:11<00:00, 87.43it/s, avg_mae=0.01620, avg_mse=0.00058]
  Visul: 100%|██████████| 64/64 [00:01<00:00, 45.00it/s]
  ```
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/plate_1_0126.jpg "Logo plate_1_0126")
![alt text](https://github.com/Chieh-hung/CV-lab-project-1/blob/master/plate_2_0126.jpg "Logo plate_2_0126")

- Unfortunately, the results of the augmented dataset doesn't clearly imporve the performance
  - Pehaps because the loss is already too low(?
  - Or perhaps the model has already overfed with the overall dataset(?
  - Perhaps try adjust brightness for augmentation in future testing
  
