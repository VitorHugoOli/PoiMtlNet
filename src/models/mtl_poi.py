import torch
from torch import nn

from configs.globals import DEVICE
from models.npc import NPC
from models.pcat import PoiCat


class MTLPOI(nn.Module):
    def __init__(self, input_dim, shared_layer_size, num_classes, num_heads, num_layers, seq_length, num_shared_layers):
        super(MTLPOI, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embedding = torch.nn.Embedding(1, input_dim)

        shared_linear_layers = []
        shared_linear_layers.append(nn.Linear(input_dim, shared_layer_size))
        shared_linear_layers.append(nn.LeakyReLU())
        shared_linear_layers.append(nn.Dropout())

        for _ in range(num_shared_layers - 1):
            shared_linear_layers.append(nn.Linear(shared_layer_size, shared_layer_size))
            shared_linear_layers.append(nn.LeakyReLU())
            shared_linear_layers.append(nn.Dropout())

        self.shared_layers = nn.Sequential(*shared_linear_layers)

        self.pcat = PoiCat(shared_layer_size, num_classes)
        self.npc = NPC(shared_layer_size, num_classes, num_heads, seq_length, num_layers)

    def forward(self, x1, x2):
        idxs = x2.sum(-1) == 0

        if torch.any(idxs):
            replace_tensor = self.embedding(torch.tensor(0, dtype=torch.long).to(DEVICE))
            x2[idxs] = replace_tensor

        shared_output1 = self.shared_layers(x1)
        shared_output2 = self.shared_layers(x2)

        out1, r = self.pcat(shared_output1)

        out2 = self.npc(shared_output2)

        return out1, r, out2

    def forward_nextpoi(self, x):
        idxs = x.sum(-1) == 0
        x[idxs] = self.embedding(torch.tensor(0, dtype=torch.long).to(DEVICE))

        shared_output = self.shared_layers(x)

        out = self.npc(shared_output)

        return out

    def forward_categorypoi(self, x):
        shared_output = self.shared_layers(x)

        out, r = self.pcat(shared_output)

        return out, r
