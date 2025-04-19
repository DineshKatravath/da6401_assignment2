# importing libraries
import os
import numpy as np
import random
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import wandb
from kaggle_secrets import UserSecretsClient
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.functional as F
import gc
import matplotlib.pyplot as plt
from io import BytesIO
import argparse


activation_functions = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'silu': torch.nn.SiLU,
    'mish': None  # Special case for Mish (since it's not built-in)
}

def get_activation_fn(config_activation_fn):
    # Handled 'mish' as a special case (it's not a built-in PyTorch activation function)
    if config_activation_fn.lower() == 'mish':
        class Mish(torch.nn.Module):
            def forward(self, x):
                return x * torch.tanh(torch.nn.functional.softplus(x))
        return Mish
    # Otherwise, used the dictionary for built-in activations
    elif config_activation_fn.lower() in activation_functions:
        return activation_functions[config_activation_fn.lower()]
    else:
        raise ValueError(f"Activation function {config_activation_fn} not recognized.")

# Custom Dataset for iNaturalist

class iNaturalistDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(data_dir))

        # storing all labels, iamge paths
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    # get a particular image
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(batch_size, data_augmentation=True):
    """
    Load and prepare the iNaturalist dataset with data augmentation and parallelized data loading.
    """
    # Get image transformation pipeline
    transform = get_transforms(data_augmentation)

    # Load dataset
    dataset = iNaturalistDataset('/kaggle/input/inaturalist_12K/train', transform)

    # Split into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers=0
    if torch.cuda.is_available():
        num_workers = min(4, os.cpu_count())
    else:
        num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

    return train_loader, val_loader


# Data Preprocessing and Augmentation
def get_transforms(data_augmentation=True):
    if data_augmentation:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class SimpleCNN(pl.LightningModule):
    def __init__(self, convFilters, filterSizes, activationFn, useBatchNorm, dropoutRate,
                 optimizerType, learningRate, weightDecay, denseFeatures, numClasses=10):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access and logging

        # Define the loss function here
        self.lossFn = nn.CrossEntropyLoss()

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_accuracy=[]
        self.test_loss=[]
        
        self.convBlocks = nn.ModuleList() 
        in_channels = 3  

        # Construct convolutional blocks based on given filters and kernel sizes
        for out_channels, k in zip(convFilters, filterSizes):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=k // 2)]
            if useBatchNorm:
                layers.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm if enabled
            layers.append(activationFn())  # Add activation function (ReLU, GELU, etc.)
            layers.append(nn.MaxPool2d(2)) 
            if dropoutRate > 0:
                layers.append(nn.Dropout2d(dropoutRate))
            self.convBlocks.append(nn.Sequential(*layers)) 
            in_channels = out_channels  # Set input channels for next block

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            for block in self.convBlocks:
                dummy_input = block(dummy_input)
            self.flattened_size = dummy_input.view(1, -1).shape[1]  
        # Fully connected (dense) layer
        self.fc = nn.Linear(self.flattened_size, denseFeatures)
        self.fc_activation = activationFn()  # Activation after dense layer
        self.dropout = nn.Dropout(dropoutRate) if dropoutRate > 0 else nn.Identity()  # Optional dropout
        self.outputLayer = nn.Linear(denseFeatures, numClasses)  # Final classification layer

        # Cross-entropy loss for multi-class classification
        self.lossFn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Pass input through convolutional blocks
        for block in self.convBlocks:
            x = block(x)
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layer
        x = self.fc_activation(self.fc(x))  # Apply dense layer and activation
        x = self.dropout(x)  # Apply dropout (if any)
        return self.outputLayer(x)  # Output raw class scores (logits)
        
    def training_step(self, batch, batchIdx):
        # Training logic: forward pass + compute loss
        images, labels = batch
        logits = self(images)
        loss = self.lossFn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        return {'loss': loss,'acc':acc}

    def validation_step(self, batch, batchIdx):
        # Validation logic: compute loss and accuracy
        images, labels = batch
        logits = self(images)
        loss = self.lossFn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        return {'loss': loss,'acc':acc}

    def test_step(self, batch, batch_idx):
        # Validation logic: compute loss and accuracy
        images, labels = batch
        logits = self(images)
        loss = self.lossFn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        #storing accuracies to print afterwards
        self.test_accuracy.append(acc.item())
        self.test_loss.append(loss.item())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)

    def on_test_epoch_end(self):
        #printing Final Test accuracy and loss
        avg_accuracy = sum(self.test_accuracy) / len(self.test_accuracy)
        avg_loss = sum(self.test_loss) / len(self.test_loss)
        print(f"\nFinal Test Accuracy: {avg_accuracy:.4f}")
        print(f"Final Test Loss: {avg_loss:.4f}")

    def evaluate(self, dataloader, stage="train"):
        self.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                preds = torch.argmax(logits, dim=1)
    
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)
    
        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"{stage.capitalize()} Accuracy: {accuracy:.4f}, {stage.capitalize()} Loss: {avg_loss:.4f}")

    
    # Define optimizer based on provided hyperparameters
    def configure_optimizers(self):
        if self.hparams.optimizerType == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learningRate, weight_decay=self.hparams.weightDecay)
        elif self.hparams.optimizerType == 'nadam':
            return torch.optim.NAdam(self.parameters(), lr=self.hparams.learningRate, weight_decay=self.hparams.weightDecay)
        elif self.hparams.optimizerType == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.hparams.learningRate, weight_decay=self.hparams.weightDecay)
        else:
            raise ValueError(f"{self.hparams.optimizerType} optimizer not implemented")  # Handle unknown optimizer


def train_model(bestConfig):
    # Get activation function
    activation_fn = get_activation_fn(bestConfig['activationFn'])

    # Initialize model
    model = SimpleCNN(
        convFilters=bestConfig['convFilters'],
        filterSizes=bestConfig['filterSizes'],
        activationFn=activation_fn,
        useBatchNorm=bestConfig['useBatchNorm'],
        dropoutRate=bestConfig['dropoutRate'],
        optimizerType=bestConfig['optimizerType'],
        learningRate=bestConfig['learningRate'],
        weightDecay=bestConfig['weightDecay'],
        denseFeatures=bestConfig['denseFeatures'],
        numClasses=bestConfig['numClasses']
    )

    # Load Data (no augmentation for val/test)
    train_loader, val_loader = load_data(
        batch_size=bestConfig['batchSize'],
        data_augmentation=bestConfig['dataAugmentation']
    )

    # Trainer
    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,)

    # Train and validate the model
    trainer.fit(model, train_loader, val_loader)
    val_metrics = trainer.validate(model, dataloaders=val_loader, verbose=True)

    return model,val_metrics,trainer

# Get test loader
def get_test_loader(batch_size):
    transform = get_transforms(data_augmentation=False)
    test_dataset = ImageFolder("/kaggle/input/inaturalist_12K/val", transform=transform)

    num_workers = min(os.cpu_count(), 4) if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=pin_memory)
    return test_loader

def show_predictions_grid(model, test_loader, device, class_names=None):
    model.eval()
    model.to(device)

    images_to_show = 30  # Total number of images to display

    # Convert test dataset to a list of samples (images, labels)
    all_samples = list(test_loader.dataset)

    # Get 30 random samples
    random_indices = random.sample(range(len(all_samples)), images_to_show)
    selected_samples = [all_samples[i] for i in random_indices]

    # Create a 10x3 subplot grid to show 30 images
    fig, axs = plt.subplots(10, 3, figsize=(9, 30))
    axs = axs.flatten()

    with torch.no_grad():
        for i, (image, label) in enumerate(selected_samples):
            image_tensor = image.unsqueeze(0).to(device) 

            # Get model prediction
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()

            # Move image back to CPU for visualization
            image = image.cpu()
            img_np = torchvision.utils.make_grid(image).permute(1, 2, 0).numpy()

            axs[i].imshow(img_np)

            pred_label = class_names[pred] if class_names else f"{pred}"
            true_label = class_names[label] if class_names else f"{label}"

            correct = (pred == label)
            color = 'green' if correct else 'red'

            axs[i].set_title(
                f"Pred: {pred_label}\nTrue: {true_label}",
                fontsize=10,
                color=color
            )
            axs[i].axis('off')

    fig.suptitle("Test Predictions Grid (Green = Correct, Red = Wrong)", fontsize=16, y=1.02)
    plt.tight_layout()

    image_path = "random_test_predictions_grid.png"
    plt.savefig(image_path)

    # Log to Wandb
    # wandb.init(project="da6401_Assignment2")
    # wandb.log({"Random Test Grid (10×3)": wandb.Image(image_path, caption="Color-coded 10×3 Random Grid")})
    # wandb.finish()

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN using best config")

    # Model architecture
    parser.add_argument('--convFilters', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--filterSizes', type=int, nargs='+', default=[3, 3])
    parser.add_argument('--activationFn', type=str, default='relu',
                        choices=['relu', 'gelu', 'silu', 'mish'])
    parser.add_argument('--useBatchNorm', action='store_true')
    parser.add_argument('--dropoutRate', type=float, default=0.3)
    parser.add_argument('--denseFeatures', type=int, default=256)

    # Training-related
    parser.add_argument('--optimizerType', type=str, default='adam',
                        choices=['adam', 'nadam', 'rmsprop'])
    parser.add_argument('--learningRate', type=float, default=0.001)
    parser.add_argument('--weightDecay', type=float, default=1e-4)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--maxEpochs', type=int, default=10)

    # Misc
    parser.add_argument('--dataAugmentation', action='store_true')
    parser.add_argument('--numClasses', type=int, default=10)

    return parser.parse_args()

def main():
    args = parse_args()

    # Set WandB key from Kaggle secrets
    # user_secrets = UserSecretsClient()
    # os.environ['WANDB_API_KEY'] = user_secrets.get_secret("wandb_key")

    bestConfig = {
        'convFilters': args.convFilters,
        'filterSizes': args.filterSizes,
        'activationFn': args.activationFn,
        'useBatchNorm': args.useBatchNorm,
        'dropoutRate': args.dropoutRate,
        'optimizerType': args.optimizerType,
        'learningRate': args.learningRate,
        'weightDecay': args.weightDecay,
        'batchSize': args.batchSize,
        'maxEpochs': args.maxEpochs,
        'dataAugmentation': args.dataAugmentation,
        'denseFeatures': args.denseFeatures,
        'numClasses': args.numClasses
    }

    # Train using the best config
    model, val_metrics, trainer = train_model(bestConfig)
    # Now use trainer to evaluate on test set
    test_loader = get_test_loader(batch_size=bestConfig['batchSize'])
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=True)

    # save img of 10X3 predicted grid
    class_names = test_loader.dataset.classes
    show_predictions_grid(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu',class_names=class_names)

if __name__ == '__main__':
    main()
