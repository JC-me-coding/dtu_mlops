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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    batch_size = 64
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()

    test_data = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    true,all = 0,0
    
    model.eval()
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            outputs = model(images.float())
            _, predicted = torch.max(outputs.data, 1)
            all += labels.size(0)
            true += (predicted == labels).sum().item()
    print('Acc: ', true/all*100)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()