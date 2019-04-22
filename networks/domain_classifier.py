import torch
import torch.nn as nn


class GradReverse(nn.function.Function):
    def forward(self, x, lambd):
        self.lambd = lambd
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lamdb)

def grad_reverse(x, lamdb):
    return GradReverse()(x, lamdb)


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 1024) 
        self.fc2 = nn.Linear(1024, 1)
        self.drop = nn.Dropout2d(0.25)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, lambd=0.5):
        x = grad_reverse(x, lambd)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)
