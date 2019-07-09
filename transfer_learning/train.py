import torch
from torch import nn
from torchvision import models
from torch import optim

from data_prep import *


def train_model(model, trainloader, testloader, criterion, optimizer,
                epochs=1, steps=0, running_loss=0, print_every=100):
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            print('start training ...')
            steps += 1

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % 5 == 0:
                print(f'step:{steps}...')

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}... "
                      f"Train loss: {running_loss / print_every:.3f}... "
                      f"Test loss: {test_loss / len(testloader):.3f}... "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()


def get_model():
    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 2),
                                     nn.LogSoftmax(dim=1))
    return model


if __name__ == '__main__':
    model = get_model()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    trainloader, testloader = get_data()

    train_model(model, trainloader, testloader, criterion, optimizer)

