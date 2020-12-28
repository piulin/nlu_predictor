import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class NeuralNet(nn.Module):

    def __init__(self, setup, input_size, output_size):

        super(NeuralNet, self).__init__()

        with open(setup) as f:
            config = json.load(f)

        # fully connected layers
        fcs = []
        fc_drop_outs = []
        # set up the linear layer(s).
        for cname, properties in config['fully_connected'].items():
            input = properties['input']
            output = properties['output']
            if 'drop_out' in properties:
                drop_out = float(properties['drop_out'])
            else:
                drop_out = 0.0

            fc_drop_outs.append(nn.Dropout(p=drop_out))

            fc_drop_outs.append(nn.Dropout(p=drop_out))
            # a -1 in the number of input neurons refers to the input size.
            if input == -1:
                input = input_size
                # fcs.append ( nn.Linear(input_size,output) )

            # a -1 in the number of input neurons refers to the input size.
            if output == -1:
                output = output_size

            fcs.append(nn.Linear(input, output))

        self.linears = nn.ModuleList(fcs)
        self.fc_drop = nn.ModuleList(fc_drop_outs)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        # print(x.shape)
        for layer, drop in zip(self.linears[:-1], self.fc_drop[:-1]):
            # print(f"layer x: {x}")
            x = F.relu(drop(layer(x)))

        x = self.fc_drop[-1](self.linears[-1](x))
        return self.softmax(x)