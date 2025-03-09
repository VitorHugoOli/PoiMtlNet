from torch import nn


class PoiCat(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PoiCat, self).__init__()

        self.hidden_layer1 = 1024
        self.hidden_layer2 = 512
        self.hidden_layer3 = 256
        self.latent_space = 128

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_layer1),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer1, self.hidden_layer2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer2, self.hidden_layer3),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer3, self.latent_space),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space, self.hidden_layer3),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer3, self.hidden_layer2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer2, self.hidden_layer1),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer1, input_dim),
            nn.LeakyReLU(),
            nn.Linear(256, 100)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_space, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        out = self.classifier(latent)
        return out, reconstructed
