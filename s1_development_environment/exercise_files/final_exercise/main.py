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
    batch_size = 64
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    epochs = 10
    model.train()

    for epoch in range(epochs):
        for i, data in enumerate(train_dl):
            image,label = data
            optimizer.zero_grad()
            output = model(image.float())
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

    torch.save(model, 'mnist_model.pt')
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    test_dl = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    
    total_loss = 0.0
    total_correct = 0

    for data in test_dl:
        inputs, labels = data
        outputs = model(inputs.float())
        
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        prediction = outputs.argmax(dim=1, keepdim=True)
        
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        total_correct += correct

    avg_loss = total_loss / len(test_dl)
    accuracy = 100.0 * total_correct / len(test_set)

    print(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()