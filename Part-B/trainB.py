#importing all libraries
import os
import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import ResNet50_Weights
import argparse
from pytorch_lightning.callbacks import TQDMProgressBar


# Freeze all layers in the model
def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False

# Unfreeze last `k` parameters of the model
def unfreeze_last_k(model, k):
    count = 0
    for name, param in reversed(list(model.named_parameters())):
        if count < k:
            param.requires_grad = True
            count += 1
        else:
            param.requires_grad = False

# Prepare train and validation data loaders
def get_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder("/kaggle/input/inatural-12k/inaturalist_12K/train", transform=transform_train)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, [train_len, val_len])
    val_data.dataset.transform = transform_val

    num_workers = min(4, os.cpu_count()) if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

# LightningModule for ResNet50 fine-tuning
class ResNet50Finetuner(pl.LightningModule):
    def __init__(self, learning_rate, dropout, k):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT) #pretrained weights

        # Modify the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.model.fc.in_features, 10)
        )
        # Freeze all layers and unfreeze last k layers
        freeze_all(self.model)
        unfreeze_last_k(self.model, k)
        
        self.criterion = nn.CrossEntropyLoss()
        self.test_accuracy=[]
        self.test_loss=[]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.test_accuracy.append(acc.item())
        self.test_loss.append(loss.item())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)

    def on_test_epoch_end(self):
        avg_accuracy = sum(self.test_accuracy) / len(self.test_accuracy)
        avg_loss = sum(self.test_loss) / len(self.test_loss)
        print(f"\nFinal Test Accuracy: {avg_accuracy:.4f}")
        print(f"Final Test Loss: {avg_loss:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.learning_rate
        )

# Get test loader
def get_test_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_dataset = ImageFolder("/kaggle/input/inatural-12k/inaturalist_12K/val", transform=transform)

    num_workers = min(os.cpu_count(), 4) if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=pin_memory)
    return test_loader

def train(best_config):
    model = ResNet50Finetuner(learning_rate=best_config['learning_rate'],dropout=best_config['dropout'],k=best_config['k'])

    train_loader, val_loader = get_data_loaders(best_config['batch_size'])
    
    trainer = pl.Trainer(max_epochs=best_config['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        callbacks=[TQDMProgressBar(refresh_rate=1)])
    
    trainer.fit(model, train_loader, val_loader)
    
    test_loader = get_test_loader(best_config['batch_size'])
    trainer.test(model, test_loader)

# CLI parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on iNaturalist")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout before final layer")
    parser.add_argument("--k", type=int, default=20, help="Number of trainable layers from the end")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)