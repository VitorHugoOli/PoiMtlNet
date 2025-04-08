import torch
from torch import nn

from configs.globals import DEVICE
from models.support.utils import MultiHeadCrossAttention


class CategoryNet(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super(CategoryNet, self).__init__()

        # Modern CNN architecture with residual connections
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive to handle various input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate feature map size after convolutions
        feature_size = 64 * 4 * 4

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Cross-attention for task interaction
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads=4, dropout=dropout)

        # Projection to prepare for CNN
        self.projection = nn.Linear(embed_dim, 22 * 22)

    def forward(self, x, context=None):
        batch_size, seq_len, dim = x.shape

        # Apply cross-attention if context is provided
        if context is not None:
            x = x + self.cross_attention(x, context)

        # Project and reshape for CNN
        x = self.projection(x)
        x = x.view(batch_size * seq_len, 1, 22, 22)  # Reshape for CNN

        # Apply CNN blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        if DEVICE.type == "mps":
            x = x.to("cpu")
            x = self.adaptive_pool(x)
            x = x.to(DEVICE)
        else:
            x = self.adaptive_pool(x)

        # Flatten and classify
        x = x.view(batch_size * seq_len, -1)
        x = self.classifier(x)

        return x

    @staticmethod
    def reshape_output(pred, truth):
        """
        Reshape the output of the model to match the ground truth shape.
        :param pred: Model predictions
        :param truth: Ground truth labels
        :return: Reshaped predictions
        """

        # Reshape the predictions to match the ground truth shape
        return pred, truth.view(-1)
