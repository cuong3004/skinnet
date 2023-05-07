from glob import glob 
from PIL import Image 
import torchvision.transforms as transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pytorch_lightning as pl
from model import get_mobile_former, get_skinnet_v1, get_skinnet_v2, get_skinnet_v3
import torch.nn as nn 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F

jitter_strength = 0.8
img_size = 128

normalize=transforms.Normalize(IMAGENET_DEFAULT_MEAN, 
                                IMAGENET_DEFAULT_STD)
final_transform = transforms.Compose([transforms.ToTensor(), normalize])

transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(128, scale=(0.5,0.5)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.2 * jitter_strength,
                ),
                final_transform
            ]
        )

transform_valid = transforms.Compose(
            [
                final_transform
            ]
        )

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

class ImgFolderHam:
    def __init__(self, list_file, transforms=None):
        self.list_file = list_file
        self.transforms = transforms
    def __len__(self):
        return len(self.list_file)
    def __getitem__(self, idx):
        img = Image.open(self.list_file[idx])
        img = img.resize((img_size,img_size))
        if self.transforms:
            img = self.transforms(img)
        # img = np.asarray(img)

        label  = label_dir[self.list_file[idx].split("/")[-2]]
        return img, label


list_file_ham_train = glob(f"ham_data_jpg/*/*.jpg")
data_ham_train = ImgFolderHam(list_file_ham_train)

class SkinDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = './'):
        super().__init__()
        self.batch_size = batch_size

        self.transform_train = transform_train

        self.transform_valid = transform_valid
        self.transform_test = transform_valid
        
        self.num_classes = 2

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            list_file_ham = glob(f"ham_data_jpg/*/*.jpg")
            
            list_file_ham_train, list_file_ham_test = train_test_split(list_file_ham ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True) 
            
            self.data_train = ImgFolderHam(list_file_ham_train, self.transform_train)
            self.data_val = ImgFolderHam(list_file_ham_test, self.transform_test)
            self.data_test = ImgFolderHam(list_file_ham_test, self.transform_test)

        # self.train_dataloader
        
        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
    
class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x



from torchmetrics.functional import accuracy, precision, recall, f1_score
average = "macro"

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        model_cls = get_mobile_former()
        model_cls.classifier.classifier[1] = IdentityLayer()
        
        model_cls.classifier.classifier[1] = nn.Linear(1024, 7)

        self.model = model_cls

        self.acc = accuracy
        self.pre = precision
        self.rec = recall
        
        self.all_preds = []
        self.all_labels = []
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        acc = self.acc(logits, y, num_classes=2)

        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)

        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        
        pred = logits.argmax(dim=1)
        
        self.all_preds.append(pred.to('cpu'))
        self.all_labels.append(y.to('cpu'))

    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        # print(all_preds.shape)
        acc = accuracy(all_preds, all_labels, task="multiclass")
        pre = precision(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        rec = recall(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        f1 = f1_score(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        
        self.log('val_acc', acc)
        self.log('val_pre', pre[1])
        self.log('val_rec', rec[1])
        self.log('val_f1', f1[1])
        
        self.all_preds = []
        self.all_labels = []
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        
        
        pred = logits.argmax(dim=1)
        
        self.all_preds.append(pred.to('cpu'))
        self.all_labels.append(y.to('cpu'))
    
    def on_test_epoch_end(self):
        
        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        
        acc = accuracy(all_preds, all_labels, task="multiclass")
        pre = precision(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        rec = recall(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        f1 = f1_score(all_preds, all_labels, task="multiclass", average=average, num_classes=7)
        
        self.log('test_acc', acc)
        self.log('test_pre', pre[1])
        self.log('test_rec', rec[1])
        self.log('test_f1', f1[1])
        
        self.all_preds = []
        self.all_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    
# class Lit(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
        
#         model_cls = get_mobile_former()
#         model_cls.classifier.classifier[1] = IdentityLayer()
        
#         model_cls.classifier.classifier[1] = nn.Linear(1024, 7)
        
#     def 
        
dm = SkinDataModule(batch_size=32)

model_lit = LitModel()

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max')

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="byol_fine_tune_12", name="skin_vit_new_1", log_model="all")


# Initialize a trainer
trainer = pl.Trainer(max_epochs=100,
                     gpus=1, 
                    #  step-
                    # limit_train_batches=0.3,
                     logger=wandb_logger,
                     callbacks=[
                        checkpoint_callback],
                     )

# Train the model ⚡🚅⚡
trainer.fit(model_lit, dm)

model_lit = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
# model_lit.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trainer.test(model_lit, dm)
# print(trainer.checkpoint_callback.best_model_path)

