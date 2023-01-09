import argparse
import sys

import torch
import click
import torch.optim as optim

from data import mnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np
@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_set, _ = mnist()
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
    trainingloss = []
    xaxislol = np.arange(1,21)
    model.train()
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs,labels = data
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        trainingloss.append(running_loss/25000)
    
    torch.save(model, 'trained_model.pt')
    plt.plot(xaxislol,trainingloss)
    plt.ylabel('training loss')
    plt.xlabel('epoch')
    plt.savefig('training.png')
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, num_workers=4)
    correct = 0
    tot = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            tot += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: ', correct/tot*100)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()