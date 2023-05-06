import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from lightly.data import LightlyDataset, SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from functools import partial
from typing import Sequence, Tuple, Union
from glob import glob
# import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
# from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18
from torchvision.utils import make_grid
from mobile_former import mobile_former_26m
from PIL import Image
from transformers import get_constant_schedule_with_warmup
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

batch_size = 128
num_workers = 2  # to run notebook on CPU
max_epochs = 200

class BarlowTwinsTransform:
    def __init__(self, train=True, input_height=224, 
                        gaussian_blur=True, jitter_strength=1.0, 
                        normalize=transforms.Normalize(IMAGENET_DEFAULT_MEAN, 
                                                        IMAGENET_DEFAULT_STD)):
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.normalize = normalize
        self.train = train

        color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        color_transform = [transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            color_transform.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_height, scale=(0.5,0.5)),
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform,
            ]
        )

        self.finetune_transform = None
        if self.train:
            self.finetune_transform = transforms.Compose(
                [
                    transforms.RandomCrop(self.input_height, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, sample):
        return self.transform(sample), self.transform(sample), self.finetune_transform(sample)


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class BYOL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # resnet = torchvision.models.resnet18()
        mobileformer = mobile_former_26m()
        
        state_dict = torch.load("mobile-former-26m.pth",  map_location=torch.device('cpu'))["state_dict"]

        mobileformer.load_state_dict(state_dict)
        mobileformer.classifier.classifier[1] = IdentityLayer()
        self.backbone = mobileformer
        self.projection_head = BYOLProjectionHead(1024, 512, 256)
        self.prediction_head = BYOLPredictionHead(256, 512, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.linear = nn.Linear(1024, 7)

        self.criterion = NegativeCosineSimilarity()

        # out = mobileformer(torch.ones((2,3,64,64)))
        # print(out.shape)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        # print(y.shape)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p, y

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    
    def training_step(self, batch, batch_idx):
        # momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
        # (x0, x1), labels, _ = batch
        (x0, x1, finetune_view), labels = batch
        p0, _ = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1, _ = self.forward(x1)
        z1 = self.forward_momentum(x1)

        # y = torch.cat([y0,y1], 0)
        loss2 = 0
        if batch_idx % 2 == 0:
            with torch.no_grad():
                feats = self.backbone(finetune_view).flatten(start_dim=1)
            preds_linear = self.linear(feats.detach()) 
            # labels_new = labels.repeat(2)
            loss_linear = F.cross_entropy(preds_linear, labels)
            loss2 += loss_linear*0.03

        loss1 = 1+ 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        
        loss =  loss1 + loss2
        # acc = self.accuracy(preds_linear, labels)
        # self.log("acc", acc[0], prog_bar=True)
        self.log("loss1", loss1, prog_bar=True)
        self.log("loss2", loss2, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        x, labels = batch
        
        feats = self.backbone(x).flatten(start_dim=1)
        preds_linear = self.linear(feats) 
        # labels_new = labels.repeat(2)
        loss= F.cross_entropy(preds_linear, labels)

        acc = self.accuracy(preds_linear, labels)
        self.log("acc", acc[0], prog_bar=True, on_epoch=True)
        # self.log("loss", loss)
        return loss
    
    @torch.no_grad()
    def accuracy(self, preds, targets, k=(1,5)):
        preds = preds.topk(max(k), 1, True, True)[1].t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))

        res = []
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = get_constant_schedule_with_warmup(optimizer, 
                        num_warmup_steps = 20, # Default value in run_glue.py
                        )
        return [optimizer], [scheduler]
    

img_size = 64

class ImgFolderIs:
    def __init__(self, list_file):
        self.list_file = list_file
    def __len__(self):
        return len(self.list_file)
    def __getitem__(self, idx):
        img = Image.open(self.list_file[idx])
        img = img.resize((img_size,img_size))
        # img = np.asarray(img)
        return img

list_file_is = glob(f"is_data_jpg/*.jpg")
data_is = ImgFolderIs(list_file_is)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
label_dir = {k:v for v,k in enumerate(lesion_type_dict.keys())}
label_dir_invert = {k:v for k,v in enumerate(lesion_type_dict.keys())}

bl_transform = BarlowTwinsTransform(input_height=img_size, jitter_strength=0.8)


class ImgFolderHam:
    def __init__(self, list_file):
        self.list_file = list_file
    def __len__(self):
        return len(self.list_file)
    def __getitem__(self, idx):
        img = Image.open(self.list_file[idx])
        img = img.resize((img_size,img_size))
        img = bl_transform.finetune_transform(img)
        # img = np.asarray(img)

        label  = label_dir[self.list_file[idx].split("/")[-2]]
        return img, label

list_file_ham = glob(f"ham_data_jpg/*/*.jpg")
data_ham = ImgFolderHam(list_file_ham)
# data_ham[0][0].shape

len_data_train = int(len(data_ham)*0.6)
len_data_valid = len(data_ham) - len_data_train

data_ham_train, data_ham_valid = torch.utils.data.random_split(data_ham, [len_data_train, len_data_valid])


dataloader_ham_train = torch.utils.data.DataLoader(
    data_ham_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

dataloader_ham_valid = torch.utils.data.DataLoader(
    data_ham_valid,
    batch_size=batch_size,
    drop_last=True,
    num_workers=0,
)



import itertools

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
# dataloader_iter = iter(itertools.cycle(dataloader_ham))
dataloader_iter = cycle(dataloader_ham_train)
next(dataloader_iter)

import torch
from torchvision import transforms


def collate_fn(batch):
    # Separates the batch into separate tensors for input and target
    inputs = [item for item in batch]

    inputs_ham, labels_ham = next(dataloader_iter)

    inputs_transformed_1 = torch.stack([bl_transform.transform(img) for img in inputs])
    inputs_transformed_2 = torch.stack([bl_transform.transform(img) for img in inputs])
    
    return (inputs_transformed_1, inputs_transformed_2, inputs_ham), labels_ham


dataloader_is = torch.utils.data.DataLoader(
    data_is,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)



x_example, y_example = next(iter(dataloader_is))
x_example[0].shape


import matplotlib.pyplot as plt
def show_images(images_batch):
    # Convert tensor to numpy array
    images = images_batch[0].numpy()
    
    # Denormalize the images
    mean = np.array(IMAGENET_DEFAULT_MEAN)
    std = np.array(IMAGENET_DEFAULT_STD)
    
    
    # Transpose the image array to (batch_size, height, width, channels)
    images = images.transpose((0, 2, 3, 1)).astype(np.float32)

    images = std * images + mean
    print(images.min(), images.max(), images.dtype)
    
    # Create a grid of images
    num_images = 20
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_images:
                ax[i][j].imshow(images[i * cols + j])
                ax[i][j].axis('off')
    
    plt.show()

    images = images_batch[1].numpy()
    
    # Denormalize the images
    mean = np.array(IMAGENET_DEFAULT_MEAN)
    std = np.array(IMAGENET_DEFAULT_STD)
    
    
    # Transpose the image array to (batch_size, height, width, channels)
    images = images.transpose((0, 2, 3, 1)).astype(np.float32)

    images = std * images + mean
    print(images.min(), images.max(), images.dtype)
    
    # Create a grid of images
    num_images = 20
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_images:
                ax[i][j].imshow(images[i * cols + j])
                ax[i][j].axis('off')
    
    plt.show()

# Show the batch of images as a grid
show_images(x_example)


from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='boyl_new', name='0.99')

model = BYOL()

from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    filename="model",
    save_top_k=1,
    verbose=True,
    monitor='acc',
    mode='max',
)

from pytorch_lightning.callbacks import LearningRateMonitor
lr_monitor = LearningRateMonitor(logging_interval='step')

gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(max_epochs=max_epochs, devices=1, 
                    precision="16-mixed", logger=wandb_logger, 
                    default_root_dir="/content/drive/MyDrive/Data/log_skin_byol", callbacks=[checkpoint_callback,
                                                                                                             lr_monitor])

trainer.fit(model, train_dataloaders=dataloader_is, val_dataloaders=dataloader_ham_valid)
