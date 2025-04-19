## Assignment Outline

- In this assignment I tried to build a CNN model using pytorch.

## How to create a model ?

```
model = SimpleCNN(convFilters=convFilters,filterSizes=filterSizes,activationFn=activationFn,useBatchNorm=useBatchNorm,dropoutRate=dropoutRate,
                  optimizerType=optimizerType,learningRate=learningRate,weightDecay=weightDecay,denseFeatures=denseFeatures,numClasses=numClasses)
```

- details of the parameters are listed below

## How to train the model ?

```
train_model(config)
```

- train function takes config which has all the parameters required

# Part A

In the part A of the assignment we tried to implement the cnn model we are talking.

train_A.py is the file one needs to run to see the running of the model.

## Dataset
Ensure the iNaturalist 12K dataset is structured like:- 
```bash
/kaggle/input/inatural-12k/inaturalist_12K/train
```
or
```bash
/kaggle/input/inatural-12k/inaturalist_12K/val
```

## Features

- Dynamic CNN architecture configuration
- Optional data augmentation
- Batch normalization and dropout options
- Multiple activation and optimizer choices
- Training and validation split with optional test evaluation
- Easy integration with logging tools (e.g., Weights & Biases)

## Command Line Arguments

| Argument               | Type    | Default     | Description |
|------------------------|---------|-------------|-------------|
| `--convFilters`        | list[int] | `[32, 64]` | Number of filters in each conv layer |
| `--filterSizes`        | list[int] | `[3, 3]`    | Kernel sizes for conv layers |
| `--activationFn`       | str     | `relu`      | Activation function: `relu`, `gelu`, `silu`, or `mish` |
| `--useBatchNorm`       | flag    | `False`     | Use batch normalization after conv layers |
| `--dropoutRate`        | float   | `0.3`       | Dropout rate before dense layers |
| `--denseFeatures`      | int     | `256`       | Number of neurons in the fully connected layer |
| `--optimizerType`      | str     | `adam`      | Optimizer: `adam`, `nadam`, or `rmsprop` |
| `--learningRate`       | float   | `0.001`     | Learning rate for optimizer |
| `--weightDecay`        | float   | `1e-4`      | Weight decay (L2 regularization) |
| `--batchSize`          | int     | `64`        | Batch size for training and validation |
| `--maxEpochs`          | int     | `10`        | Number of training epochs |
| `--dataAugmentation`   | flag    | `False`     | Enable data augmentation (random crop, flip, etc.) |
| `--numClasses`         | int     | `10`        | Number of output classes |

##  Usage

You can run the training script with default or custom parameters:

```bash
python train_cnn.py
```
or
```bash
python train_cnn.py \
  --convFilters 64 128 256 \
  --filterSizes 3 3 3 \
  --activationFn gelu \
  --useBatchNorm \
  --dropoutRate 0.5 \
  --denseFeatures 512 \
  --optimizerType rmsprop \
  --learningRate 0.0005 \
  --weightDecay 1e-5 \
  --batchSize 32 \
  --maxEpochs 20 \
  --dataAugmentation \
  --numClasses 10
```

# Part B

In this we are going to use the already existed model here I am using Resnet50.

- Here I am using the Renet50 and training that.
- for this you can pass the learning rate and weight decay and check how much accuracy you are getting.

now to run the model and check its running accuracies


## Features

- Uses `torchvision.models.resnet50` with pretrained weights
- Option to fine-tune only the last `k` layers
- Dropout applied before the final classification layer
- Supports GPU training via PyTorch Lightning
- Includes logging and testing hooks
- Compatible with Weights & Biases for tracking

---

##  Parameters

| Argument        | Type   | Description                                  | Example     |
|-----------------|--------|----------------------------------------------|-------------|
| `learning_rate` | float  | Learning rate for the Adam optimizer         | `0.0003`    |
| `dropout`       | float  | Dropout rate before the final dense layer    | `0.5`       |
| `k`             | int    | Number of trainable layers from the end      | `20`        |
| `batch_size`    | int    | Batch size for training and evaluation       | `64`        |
| `epochs`        | int    | Number of training epochs                    | `10`        |

You can define these parameters in a dictionary called `best_config` and pass them to the training function.

---

## How to Use

1. Make sure you define your best config (hyperparameters):

```python
best_config = {
    "learning_rate": 0.0003,
    "dropout": 0.5,
    "k": 20,
    "batch_size": 64,
    "epochs": 10
}
```
2. call training function
   train(best_config)
