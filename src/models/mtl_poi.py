import torch
from torch import nn

from configs.globals import DEVICE
from models.category_net import CategoryPoiNet
from models.next_poi_net import NextPoiNet


class MTLnet(nn.Module):
    def __init__(self, feature_size, shared_layer_size, num_classes, num_heads, num_layers, seq_length,
                 num_shared_layers):
        super(MTLnet, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.embedding = torch.nn.Embedding(1, feature_size)

        shared_linear_layers = []  # lista de layers compartilhadas

        # add primeira camada de input (feature_size -> shared_layer_size)
        shared_linear_layers.append(nn.Linear(feature_size, shared_layer_size))
        shared_linear_layers.append(nn.LeakyReLU())
        shared_linear_layers.append(nn.Dropout())

        # add layers intermediarias
        for _ in range(num_shared_layers - 1):
            shared_linear_layers.append(nn.Linear(shared_layer_size, shared_layer_size))
            shared_linear_layers.append(nn.LeakyReLU())
            shared_linear_layers.append(nn.Dropout())

        # cria o sequential igual antes
        self.shared_layers = nn.Sequential(*shared_linear_layers)

        self.category_poi = CategoryPoiNet(num_classes)
        self.next_poi = NextPoiNet(shared_layer_size, num_classes, num_heads, seq_length, num_layers)

    def forward(self, inputs):
        category_input, next_input = inputs
        idxs = next_input.sum(-1) == 0

        if torch.any(idxs):
            replace_tensor = self.embedding(torch.tensor(0, dtype=torch.long).to(DEVICE))
            next_input[idxs] = replace_tensor

        shared_output1 = self.shared_layers(category_input)
        shared_output2 = self.shared_layers(next_input)

        out1 = self.category_poi(shared_output1)
        out2 = self.next_poi(shared_output2)

        out1 = out1.view(-1, self.num_classes)

        return out1, out2
