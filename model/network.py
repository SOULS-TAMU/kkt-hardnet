from torch import nn


class Network(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(in_features=input_dims, out_features=hidden_dims)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=hidden_dims, out_features=hidden_dims)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=hidden_dims, out_features=output_dims)

    def forward(self, input):
        x = input
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.layer3(x)
        return x
