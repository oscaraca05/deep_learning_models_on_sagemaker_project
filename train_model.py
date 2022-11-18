#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import json
import logging
import os
import sys
import io

import boto3

# PIL throws this error, if the loaded image file is truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, models, transforms

import argparse

# ====================================#
# 1. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, hook, loss_criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, loss_criterion, optimizer, hook, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    # for epoch in range(1, args.epochs + 1):
    model.train()
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # num_classes = 7
    num_classes = 133
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    hook.register_loss(loss_criterion)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    input_size = 224
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # data_dir = "s3://sagemaker-us-east-1-155431344840/deep-learning-models-on-sagemaker-project/dogImages"
    
     # transforms.Normalize((0.1307,), (0.3081,))
    
    logger.info("Get train data loader")
    training_dir = os.path.join(args.data_dir, "train/")
    train_transform = transforms.Compose([transforms.RandomResizedCrop(input_size), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.ImageFolder(training_dir, transform = train_transform)
    subset_ix = list(range(0, 200))
    # train_data = torch.utils.data.Subset(train_data, subset_ix) # for accelerating the process
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= args.batch_size, shuffle=True)
    
    logger.info("Get test data loader")
    test_dir = os.path.join(args.data_dir , "test/")
    test_transform = transforms.Compose([transforms.RandomResizedCrop(input_size), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    # test_data = torch.utils.data.Subset(test_data, subset_ix) # for accelerating the process
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= args.test_batch_size)
    
    
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimizer, hook, epoch)

        '''
        TODO: Test the model to see its accuracy
        '''
        test(model, test_loader, hook, loss_criterion)

        '''
        TODO: Save the trained model
    '''
    # torch.save(model, path)
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    # buffer = io.BytesIO()
    # torch.save(model, buffer)
    # s3 = boto3.client('s3')
    # output_model_file =  "deep-learning-models-on-sagemaker-project/pytorch_model.json"
    # s3.put_object(Bucket="sagemaker-us-east-1-155431344840", Key=output_model_file, Body=buffer.getvalue())

def model_fn(model_dir):
    logger.info("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    logger.info("loading model abc...")
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    logger.info("model abc loaded!")
    
    return model.to(device)    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
