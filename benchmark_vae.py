# intended to test pythae
import torch
import torchvision.datasets as datasets

#%load_ext autoreload
#%autoreload 2

from pythae.models import Adversarial_AE, Adversarial_AE_Config
from pythae.trainers import AdversarialTrainer, AdversarialTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist import Encoder_VAE_MNIST, Decoder_AE_MNIST
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.

config = AdversarialTrainerConfig(
    output_dir='my_model',
    learning_rate=1e-3,
    batch_size=100,
    num_epochs=100,
)


model_config = Adversarial_AE_Config(
    input_dim=(1, 28, 28),
    latent_dim=16,
    adversarial_loss_scale=0.9
)

model = Adversarial_AE(
    model_config=model_config,
    encoder=Encoder_VAE_MNIST(model_config), 
    decoder=Decoder_AE_MNIST(model_config) 
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset
)

import os
