#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# %%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from plotly_utils import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(x, t.tensor(0.0))


if MAIN:
    tests.test_relu(ReLU)

# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        sf = 1 / np.sqrt(in_features)

        weight = sf * (2 * t.rand(out_features, in_features) - 1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias = sf * (2 * t.rand(out_features) - 1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or
        # None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


if MAIN:
    tests.test_linear_parameters(Linear, bias=False)
    tests.test_linear_parameters(Linear, bias=True)
    tests.test_linear_forward(Linear, bias=False)
    tests.test_linear_forward(Linear, bias=True)

# %%

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])

# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(in_features=28 * 28, out_features=100)
        self.relu = ReLU()
        self.linear2 = Linear(in_features=100, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1(self.flatten(x))))


if MAIN:
    tests.test_mlp_module(SimpleMLP)
    tests.test_mlp_forward(SimpleMLP)

# %%

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

    # Get the first batch of test data, by starting to iterate over `mnist_testloader`
    for img_batch, label_batch in mnist_testloader:
        print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
        break

    # Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
    for img, label in mnist_testset:
        print(f"{img.shape=}\n{label=}\n")
        break

    t.testing.assert_close(img, img_batch[0])
    assert label == label_batch[0].item()

# %%

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
if MAIN:
    print(device)

# %%

if MAIN:
    model = SimpleMLP().to(device)
    
    batch_size = 128
    epochs = 3
    
    mnist_trainset, _ = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    
    for epoch in range(epochs):
        pbar = tqdm(mnist_trainloader)
    
        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
    
            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}", refresh=False)
    
    line(
        loss_list,
        x_max=epochs * len(mnist_trainset),
        labels={"x": "Examples seen", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )

# %%

@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains & returns the model, using training parameters from the `args` object. Returns the model,
    and loss list.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, _ = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)

        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}", refresh=False)

    return loss_list, model


if MAIN:
    args = SimpleMLPTrainingArgs()
    loss_list, model = train(args)

    line(
        loss_list,
        x_max=args.epochs * len(mnist_trainset),
        labels={"x": "Examples seen", "y": "Cross entropy loss"},
        title="SimpleMLP training on MNIST",
        width=700,
    )

# %%

def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], list[float], SimpleMLP]:
    """
    Trains the model, using training parameters from the `args` object.

    Returns:
        The model, and lists of loss & accuracy.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_list = []
    accuracy_list = []
    accuracy = 0.0

    for epoch in range(args.epochs):
        # Training loop
        pbar = tqdm(mnist_trainloader)
        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}", refresh=False)

        # Validation loop
        num_correct_classifications = 0
        for imgs, labels in mnist_testloader:
            # Move data to device, perform forward pass in inference mode
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            # Compute num correct by comparing argmaxed logits to true labels
            predictions = t.argmax(logits, dim=1)
            num_correct_classifications += (predictions == labels).sum().item()

        # Compute & log total accuracy
        accuracy = num_correct_classifications / len(mnist_testset)
        accuracy_list.append(accuracy)

    return loss_list, accuracy_list, model


if MAIN:
    args = SimpleMLPTrainingArgs()
    loss_list, accuracy_list, model = train(args)

    line(
        y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
        use_secondary_yaxis=True,
        x_max=args.epochs * len(mnist_trainset),
        labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
        title="SimpleMLP training on MNIST",
        width=800,
    )

# %%

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_height = kernel_width = kernel_size
        sf = 1 / np.sqrt(in_channels * kernel_width * kernel_height)
        self.weight = nn.Parameter(sf * (2 * t.rand(out_channels, in_channels, kernel_height, kernel_width) - 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


if MAIN:
    tests.test_conv2d_module(Conv2d)
    m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

# %%

class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x

# %%

class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
          variance = t.var(x, unbiased = False, dim = [0, 2, 3])
          variance_un = t.var(x, unbiased = True, dim = [0, 2, 3])
          mean = t.mean(x, dim = [0, 2, 3])

          self.running_mean = self.momentum * mean + (1-self.momentum)* self.running_mean
          self.running_var = self.momentum * variance_un + (1-self.momentum) * self.running_var

        else:
          variance = self.running_var
          mean = self.running_mean

        self.num_batches_tracked += 1

        reshape = lambda x: einops.rearrange(x, "c -> 1 c 1 1")

        x_normalized = (x-reshape(mean)) / (reshape(variance) + self.eps).sqrt()

        x_affined = x_normalized * reshape(self.weight) + reshape(self.bias)

        return x_affined



        # raise NotImplementedError()

    def extra_repr(self) -> str:
        return ",".join([f"{key} = {getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])
        # raise NotImplementedError()


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)
# %%
class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """

        average_x = t.mean(x, dim = (2, 3))
        return average_x
        # raise NotImplementedError()

if MAIN:
    tests.test_averagepool(AveragePool)
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity

        # 第二层CNN的in_channels是第一层的out
        self.left = nn.Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=3, stride=first_stride, padding= 1),
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(in_channels=out_feats, out_channels=out_feats, kernel_size=3, stride=1, padding= 1),
            BatchNorm2d(num_features=out_feats)
        )
        # 命名要规范。。？
        self.option = nn.Sequential(
            nn.Identity()
            if is_shape_preserving
            else Sequential(
            Conv2d(in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=first_stride),
            BatchNorm2d(num_features=out_feats))
        )

        self.relu = ReLU()

        # if (is_shape_preserving):
        #     self.net = nn.Sequential(
        #         left + option,
        #         ReLU()
        #     )
        # else:
        #     self.net = nn.Sequential(
        #         left,
        #         ReLU()
        #     )


        # raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """
        # raise NotImplementedError()
        return self.relu(self.left(x) + self.option(x))

if MAIN:
    tests.test_residual_block(ResidualBlock)
# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()
        # YOUR CODE HERE - define all components of block group
        # raise NotImplementedError()
        blocks = [
        ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)
        ] + [
            ResidualBlock(in_feats=out_feats, out_feats=out_feats, first_stride=1) # 注意：后续 block 的 stride 通常为 1
            for _ in range(1, n_blocks)
        ]

        self.model = Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.model(x)

if MAIN:
    tests.test_block_group(BlockGroup)  
# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        # YOUR CODE HERE - define all components of resnet34
        # CNN要区分kernal，channel，height，weight，feature_map大小（(H+2*padding-k/2)+1）
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=out_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(num_features=out_feats0), # (H+2*padding-k/2)+1
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            BlockGroup(n_blocks=self.n_blocks_per_group[0], in_feats=out_feats0, out_feats=self.out_features_per_group[0], first_stride=self.first_strides_per_group[0]),
            *[BlockGroup(n_blocks = self.n_blocks_per_group[_], in_feats = self.out_features_per_group[_-1], out_feats = self.out_features_per_group[_], first_stride = self.first_strides_per_group[_] ) for _ in range(1,4) ],
            AveragePool(),
            Linear(in_features=512, out_features=self.n_classes)
        )
        # raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        # raise NotImplementedError()
        return self.model(x)

if MAIN:
    my_resnet = ResNet34()

    # (1) Test via helper function `print_param_count`
    target_resnet = models.resnet34()  # without supplying a `weights` argument, we just initialize with random weights
    utils.print_param_count(my_resnet, target_resnet)

    # (2) Test via `torchinfo.summary`
    print("My model:", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
    print(
        "\nReference model:",
        torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2),
        sep="\n",
    )

# %%
def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters &
    # buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the
    # pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)
print("Weights copied successfully!")

# %%
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

# %%
display(images[0])
# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0).to(device)
assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %%
@t.inference_mode()
def predict(
    model: nn.Module, images: Float[Tensor, "batch rgb h w"]
) -> tuple[Float[Tensor, " batch"], Int[Tensor, " batch"]]:
    """
    Returns the maximum probability and predicted class for each image, as a tensor of floats and
    ints respectively.
    """
    model.eval()
    # raise NotImplementedError()
    # logits = model(images)  # (batch,classes:float)
    # ans_p = list()
    # ans_class = list()
    # for batch in logits:
    #     max_p = 0
    #     max_idx = 0
    #     for idx, c in logits[batch]:
            
    #         if (c > max_p):
    #             max_idx = idx
    #             max_p = c
    #     ans_p.append(max_p)
    #     ans_class.append(max_idx)
    # return (Tensor(ans_p), Tensor(ans_class))

    # 1. 计算未经归一化的 logits
    logits = model(images)
    
    # 2. 将 logits 转换为类别概率分布 (在最后一个维度 dim=-1/dim=1 上计算)
    probs = t.softmax(logits, dim=-1)
    
    # 3. 沿类别维度求最大概率值和对应的索引类别
    max_probs, max_classes = t.max(probs, dim=-1)
    
    return max_probs, max_classes



with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_probs, my_predictions = predict(my_resnet, prepared_images)
pretrained_probs, pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert (my_predictions == pretrained_predictions).all()
t.testing.assert_close(my_probs, pretrained_probs, atol=5e-4, rtol=0)  # tolerance of 0.05%
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for i, img in enumerate(images):
    table = Table("Model", "Prediction", "Probability")
    table.add_row("My ResNet", imagenet_labels[my_predictions[i]], f"{my_probs[i]:.3%}")
    table.add_row(
        "Reference Model",
        imagenet_labels[pretrained_predictions[i]],
        f"{pretrained_probs[i]:.3%}",
    )
    rprint(table)
    display(img)

# %%
class NanModule(nn.Module):
    """
    Define a module that always returns NaNs (we will use hooks to identify this error).
    """

    def forward(self, x):
        return t.full_like(x, float("nan"))


def hook_check_for_nan_output(module: nn.Module, input: tuple[Tensor], output: Tensor) -> None:
    """
    Hook function which detects when the output of a layer is NaN.
    """
    if t.isnan(output).any():
        raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
    """
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    """
    module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
    """
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()


# Create our model with a NaN in the middle, and apply a hook fn to it which checks for NaNs
model = nn.Sequential(nn.Identity(), NanModule(), nn.Identity())
model = model.apply(add_hook)

# Run the model, and our hook function should raise an error that gets caught by the try-except
try:
    input = t.randn(3)
    output = model(input)
except ValueError as e:
    print(e)

# Remove hooks at the end
model = model.apply(remove_hooks)

# %%

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

def pathological_curve_loss(x: Tensor, y: Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss, min_points=[(0, "y_min")])




# %%
def opt_fn_with_sgd(
    fn: Callable, xy: Float[Tensor, "2"], lr=0.001, momentum=0.98, n_iters: int = 100
) -> Float[Tensor, "n_iters_plus_1 2"]:
    """
    Optimize a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters+1, 2). The (x, y) values, from initial values to values after step `n_iters`.
    """
    # Make sure tensor has requires_grad=True, otherwise it can't be optimized (more on this tomorrow!)
    assert xy.requires_grad

    # raise NotImplementedError()
    optimizer = t.optim.SGD((xy,), lr=lr, momentum=momentum)
    xy_lists= list()
    xy_lists.append(xy.detach().clone())
    for i in range(n_iters):
        optimizer.zero_grad()
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        xy_lists.append(xy.detach().clone())

    return t.stack(xy_lists)



points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])


# %%
class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        self.params = list(params)  # turn params into a list (it might be a generator, so iterating over it empties it)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        self.b = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm."""
        # raise NotImplementedError()
        for b, param in zip(self.b, self.params):
            if param.grad is not None:
                g = param.grad + self.lmda * param
                if self.mu:
                    b.copy_(b * self.mu + g)
                    g = b
                param -= self.lr * g


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)


# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
        """
        self.params = list(params)  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.b = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        # raise NotImplementedError()

        for p, b, v in zip(self.params, self.b, self.v):

            g = p.grad
            if g is not None:
                # g += self.lmda * p  不能直接原地修改grad
                g=  g+self.lmda * p
                v.copy_(self.alpha * v + (1-self.alpha)*g*g)
                # 你的理解：把动量当成了原始力（Force）的累积，结果导致自适应阻尼（$\sqrt{v}$）错误地作用在了历史累积的所有物理量上。标准的定义：动量是有效步长（Normalized Velocity）的平滑。先用 RMS 消除陡度影响，再用 Momentum 保留历史速度，这是几乎所有现代化优化器
                # b.copy_(self.mu * b + g)
                # g = b
                # p -= self.lr * g/t.sqrt(v + self.eps)
                b.copy_(self.mu * b + g/(t.sqrt(v)+self.eps))
                g = b
                p -= self.lr * g

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"
        )


tests.test_rmsprop(RMSprop)


# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        # raise NotImplementedError()
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is not None:
                g = p.grad + self.lmda*p
                import math
                # m.copy_((self.beta1 * m + (1 - self.beta1) * g))
                # v.copy_((self.beta2 * v + (1-self.beta2) * m / (t.sqrt(g*g)+self.eps)) )

                # m_t = m/(1-math.pow(self.beta1, self.t))
                # v_t = ((self.beta2 * v + (1-self.beta2)*m_t)/ (t.sqrt(g*g)+self.eps))/(1-math.pow(self.beta2, self.t))
                # g = v_t
                # p -= self.lr * g
                # self.t += 1

                m.copy_((self.beta1 * m + (1 - self.beta1) * g))
                v.copy_((self.beta2 * v + (1 - self.beta2) * g*g ) )

                m_t = m/(1-math.pow(self.beta1, self.t))
                v_t = v/(1-math.pow(self.beta2, self.t))
                g = m_t/(t.sqrt(v_t)+self.eps)
                p -= self.lr * g
            # self.t += 1

        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)


# %%
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements AdamW.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for theta, m, v in zip(self.params, self.m, self.v):
            g = theta.grad
            theta *= 1 - self.lr * self.lmda
            m.copy_(self.beta1 * m + (1 - self.beta1) * g)
            v.copy_(self.beta2 * v + (1 - self.beta2) * g.pow(2))
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            theta -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


# %%
def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = 0.5 * ((x - x_mean) ** 2) / (x_sig**2)
    y_exp = 0.5 * ((y - y_mean) ** 2) / (y_sig**2)
    return norm * t.exp(-x_exp - y_exp)


means = [(1.0, -0.5), (-1.0, 0.5), (-0.5, -0.8)]


def neg_trimodal_func(x, y):
    """
    This function has 3 global minima, at `means`. Unstable methods can overshoot these minima, and
    non-adaptive methods can fail to converge to them in the first place given how shallow the
    gradients are everywhere except in the close vicinity of the minima.
    """
    z = -bivariate_gaussian(x, y, x_mean=means[0][0], y_mean=means[0][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[1][0], y_mean=means[1][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[2][0], y_mean=means[2][1], x_sig=0.2, y_sig=0.2)
    return z


plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2), min_points=means)


# %%
def rosenbrocks_banana_func(x: Tensor, y: Tensor, a=1, b=100) -> Tensor:
    """
    This function has a global minimum at `(a, a)` so in this case `(1, 1)`. It's characterized by a
    long, narrow, parabolic valley (parameterized by `y = x**2`). Various gradient descent methods
    have trouble navigating this valley because they often oscillate unstably (gradients from the
    `b`-term dwarf the gradients from the `a`-term).

    See more on this function: https://en.wikipedia.org/wiki/Rosenbrock_function.
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


plot_fn(
    rosenbrocks_banana_func,
    x_range=(-2.5, 2.5),
    y_range=(-2, 4),
    z_range=(0, 100),
    min_points=[(1, 1)],
)


# %%
def opt_fn(
    fn: Callable,
    xy: Tensor,
    optimizer_class,
    optimizer_hyperparams: dict,
    n_iters: int = 100,
) -> Tensor:
    """Optimize a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, Adam or AdamW
    optimizer_hyperparams: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    """
    assert xy.requires_grad

    optimizer = optimizer_class([xy], **optimizer_hyperparams)

    xy_list = [xy.detach().clone()]  # so that we don't unintentionally modify past values in `xy_list`

    for i in range(n_iters):
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        optimizer.zero_grad()
        xy_list.append(xy.detach().clone())

    return t.stack(xy_list)


points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    (AdamW, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(
        pathological_curve_loss,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(pathological_curve_loss, min_points=[(0, "y_min")], points=points)

# %%
points = []

optimizer_list = [
    (SGD, {"lr": 0.1, "momentum": 0.5}),
    (RMSprop, {"lr": 0.1, "alpha": 0.99, "momentum": 0.5}),
    (Adam, {"lr": 0.1, "betas": (0.9, 0.999)}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([1.0, 1.0], requires_grad=True)
    xys = opt_fn(
        neg_trimodal_func,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(neg_trimodal_func, min_points=means, points=points,x_range=(-2, 2), y_range=(-2, 2))

# %%
points = []

optimizer_list = [
    (SGD, {"lr": 0.001, "momentum": 0.9}),
    # (RMSprop, {"lr": 0.002, "alpha": 0.99, "momentum": 0.8}),
    # (Adam, {"lr": 0.002, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    # (AdamW, {"lr": 0.002, "betas": (0.9, 0.999), "weight_decay": 0.005}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([1.89, 2.91], requires_grad=True)
    xys = opt_fn(
        rosenbrocks_banana_func,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(rosenbrocks_banana_func, min_points=[(1, 1)], points=points, x_range=(-2.5, 2.5), y_range=(-2, 4), z_range=(0, 100))

# %%

optimizer_list = [
    # (SGD, {"lr": 0.001, "momentum": 0.99}),
    (AdamW, {"lr": 0.1, "betas": (0.9, 0.999)}),
]

points = []
for optimizer_class, params in optimizer_list:
    xy = t.tensor([1.5, 2.5], requires_grad=True)
    xys = opt_fn(
        rosenbrocks_banana_func, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params, n_iters=500
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(
    rosenbrocks_banana_func, x_range=(-2.5, 2.5), y_range=(-2, 4), z_range=(0, 100), min_points=[(1, 1)], points=points
)

# %%
class SGD:
    def __init__(self, params, **kwargs):
        """Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        # Deal with case where we didn't supply groups, so we just make it into a single dictionary
        if not isinstance(params, (list, tuple)):
            params = [{"params": params}]

        # Make sure each group["params"] is a list of params not a generator (so we don't iterate
        # over & destroy it!)
        for p in params:
            p["params"] = list(p["params"])

        self.param_groups = []

        # YOUR CODE HERE - fill in `self.param_groups`
        # raise NotImplementedError()
        if params[0]

    def zero_grad(self) -> None:
        raise NotImplementedError()

    @t.inference_mode()
    def step(self) -> None:
        raise NotImplementedError()


tests.test_sgd_param_groups(SGD)

# %%
@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class ResNetFinetuner:
    def __init__(self, args: ResNetFinetuningArgs):
        self.args = args

    def pre_training_setup(self):
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = AdamW(
            self.model.out_layers[-1].parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )


        self.trainset = CIFAR10(train=True)
        self.testset = CIFAR10(train=False)

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""

        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        self.logged_variables["accuracy"].append(accuracy)
        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    ex_seen=f"{self.examples_seen:06}",
                    refresh=False,
                )

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.2f}",
                ex_seen=f"{self.examples_seen:06}",
            )

        return self.logged_variables

# %%
args = ResNetFinetuningArgs()
trainer = ResNetFinetuner(args)
logged_variables = trainer.train()


line(
    y=[logged_variables["loss"][: 391 * 3 + 1], logged_variables["accuracy"][:4]],
    x_max=len(logged_variables["loss"][: 391 * 3 + 1] * args.batch_size),
    yaxis2_range=[0, 1],
    use_secondary_yaxis=True,
    labels={"x": "Examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="Feature extraction with ResNet34",
    width=800,
)



# %%
@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None
    use_wandb: bool = False


class WandbResNetFinetuner(ResNetFinetuner):
    args: WandbResNetFinetuningArgs  # adding this line helps with typechecker!
    examples_seen: int = 0  # tracking examples seen (used as step for wandb)

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`."""
        super().pre_training_setup()
        # raise NotImplementedError()
        # self.run = wandb.init(entity="yuxinlin020406-arena",
        #                  project=self.args.wandb_project, 
        #                  name=self.args.wandb_name)
        wandb.init(entity="yuxinlin020406-arena",
                         project=self.args.wandb_project, 
                         name=self.args.wandb_name,
                         config = self.args,
                         mode = "online" if self.args.use_wandb else "disabled"
        )
        wandb.watch(self.model.out_layers[-1],
                    log="all",
                    log_freq=50)
        self.examples_seen = 0

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Equivalent to ResNetFinetuner.training_step, but logging the loss to wandb."""

        # raise NotImplementedError()
        # CPU -> GPU first, then perform preprocessing on GPU.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = IMAGENET_TRANSFORM(imgs)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        wandb.log({"loss": loss.item()}, step=self.examples_seen)
        # self.logged_variables["loss"].append(loss.item())
        return loss
    

    @t.inference_mode()
    def evaluate(self) -> float:
        """Equivalent to ResNetFinetuner.evaluate, but logging the accuracy to wandb."""
        # raise NotImplementedError()
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            # CPU -> GPU first, then perform preprocessing on GPU.
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = IMAGENET_TRANSFORM(imgs)

            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        # self.logged_variables["accuracy"].append(accuracy)
        # self.run.log({"accuracy": accuracy})
        wandb.log({"accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self) -> None:
        """Equivalent to ResNetFinetuner.train, but with wandb integration."""
        self.pre_training_setup()
        # raise NotImplementedError()
        
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(
                    loss=f"{loss:.3f}",
                    ex_seen=f"{self.examples_seen:06}",
                    refresh=False,
                )

            accuracy = self.evaluate()
            pbar.set_postfix(
                loss=f"{loss:.3f}",
                accuracy=f"{accuracy:.2f}",
                ex_seen=f"{self.examples_seen:06}",
            )

        wandb.finish()


args = WandbResNetFinetuningArgs(use_wandb=False)
trainer = WandbResNetFinetuner(args)
trainer.train()


# %%
import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="yuxinlin020406-arena",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()

# %%

'''

sweep_config = dict(
    method = method, # can be "grid", "random" or "bayes"
    metric = dict(
        name = metric_name, # name of the metric you're optimising (should be a numeric type logged in `wandb.log`)
        goal = goal, # either "maximize" or "minimize"
    )),
    parameters = dict(
        param_1 = dict(...),
        param_2 = dict(...),
        ...
    ),
)

parameters = dict(
    param_1 = dict(values = [...]), # uniformly sample from list of values
    param_2 = dict(values = [...], probabilities = [...]), # sample from list with given probabilities
    param_3 = dict(min = ..., max = ...), # uniform distribution over [min, max), can either be ints or floats
    param_4 = dict(min = ..., max = ..., distribution = "log_uniform_values"), # use log-uniform distribution instead
)

Hyperparameters are chosen randomly, according to the distributions given in the dictionary
Your goal is to maximize the accuracy metric
The hyperparameters you vary are:
Learning rate - a log-uniform distribution between 1e-4 and 1e-1
Batch size - sampled uniformly from (32, 64, 128, 256)
Weight decay - with 50% probability set to 0, and with 50% probability log-uniform between 1e-4 and 1e-2
'''

# YOUR CODE HERE - fill `sweep_config` so it has the requested behaviour
sweep_config = dict(
    method = "random",
    metric = dict(
        name = "metric_name",
        goal = "maximize"
    ),
    parameters = dict(
        learning_rate = dict(min = 1e-4, max = 1e-1, distribution = "log_uniform_values"),
        batch_size = dict(values = [32, 64, 128, 256]),
        weight_decay = dict(min = 1e-4, max = 1e-2, distribution = "log_uniform_values"),
        weight_bool = dict[values = [True, False]]
    ),
)


def update_args(args: WandbResNetFinetuningArgs, sampled_parameters: dict) -> WandbResNetFinetuningArgs:
    """
    Returns a new args object with modified values. The dictionary `sampled_parameters` will have
    the same keys as your `sweep_config["parameters"]` dict, and values equal to the sampled values
    of those hyperparameters.
    """
    assert set(sampled_parameters.keys()) == set(sweep_config["parameters"].keys())

    # YOUR CODE HERE - update `args` based on `sampled_parameters`
    # raise NotImplementedError()
    args.learning_rate = sampled_parameters["learning_rate"]
    args.batch_size = sampled_parameters["batch_size"]
    args.weight_decay = sampled_parameters["weight_decay"] if sampled_parameters["weight_decay_bool"] else 0.0

    return args





tests.test_sweep_config(sweep_config)
tests.test_update_args(update_args, sweep_config)

# %%
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank `src` to all other ranks.
    """
    raise NotImplementedError()


if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)

# %%
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part4_backprop"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part4_backprop.tests as tests
from part4_backprop.utils import get_mnist, visualize
from plotly_utils import line
