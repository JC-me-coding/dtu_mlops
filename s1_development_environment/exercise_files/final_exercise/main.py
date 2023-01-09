import torch
import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel


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
    train_set, _ = mnist()
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    batch_size = 64

    train_loss = []
    epochs = 25
    model.train()

    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_dl):
            image,label = data
            optimizer.zero_grad()
            output = model(image.float())
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss/25000)

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    all,true = 0,0
    
    model.eval()
    with torch.no_grad():
        for data in test_dl:
            image, label = data
            output = model(image.float())
            _, predicted = torch.max(output.data, 1)
            all += label.size(0)
            true += (predicted == label).sum().item()
    print('Acc: ', true/all*100)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()