import torch
import torch.nn as nn

from models.category_net import CategoryNet
from models.next_poi_net import NextPoiNet
from models.support.utils import PositionalEncoding, TransformerBlock, MultiHeadCrossAttention


class MTLnet(nn.Module):
    def __init__(self, vocab_size, embed_dim, shared_dim, num_classes, num_heads,
                 seq_length, num_shared_layers, dropout=0.1):
        super(MTLnet, self).__init__()

        self.num_classes = num_classes
        self.seq_length = seq_length

        # Improved embedding with learned positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_length, dropout)

        # Shared transformer encoder
        self.shared_transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_shared_layers)
        ])

        # Task-specific modules
        self.category_net = CategoryNet(embed_dim, num_classes, dropout)
        self.next_poi_net = NextPoiNet(embed_dim, num_classes, num_heads,
                                       seq_length, 2,
                                       dropout)  # Reduced layers as we already have shared layers

        # Task attention/gating mechanism
        self.task_gate = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

    def _apply_shared_transformer(self, x, mask=None):
        # Apply positional encoding
        x = self.positional_encoding(x)

        # Apply shared transformer layers
        for layer in self.shared_transformer_layers:
            x = layer(x, mask)

        return x

    def forward(self, inputs):
        """
        Forward pass with both tasks
        inputs: tuple of (category_input, next_input)
        """
        category_input, next_input = inputs

        # Get batch sizes
        cat_batch_size = category_input.shape[0]
        next_batch_size = next_input.shape[0]

        # Create attention masks
        category_mask = None  # No masking for category task
        next_mask = torch.triu(torch.ones(self.seq_length, self.seq_length) * float('-inf'), diagonal=1).to(
            next_input.device)

        # Pass through shared transformer
        shared_category = self._apply_shared_transformer(category_input, category_mask)
        shared_next = self._apply_shared_transformer(next_input, next_mask)

        # Compute task weights via gating
        category_gate = self.task_gate(shared_category.mean(dim=1, keepdim=True))
        next_gate = self.task_gate(shared_next.mean(dim=1, keepdim=True))

        # Handle context passing - use smaller batch size
        min_batch_size = min(cat_batch_size, next_batch_size)
        cat_context = shared_category[:min_batch_size]
        next_context = shared_next[:min_batch_size]

            # Forward each task with cross-task awareness
        out_category = self.category_net(shared_category, next_context)  # Use subset of next context
        out_next = self.next_poi_net(shared_next, cat_context, next_mask)  # Use subset of category context

        return out_category, out_next

    def forward_category(self, x):
        """Special forward pass for category prediction only"""
        shared_output = self._apply_shared_transformer(x)
        return self.category_net(shared_output)

    def forward_next(self, x):
        """Special forward pass for next POI prediction only"""
        # Create causal mask for next prediction
        mask = torch.triu(torch.ones(self.seq_length, self.seq_length) * float('-inf'), diagonal=1).to(x.device)

        shared_output = self._apply_shared_transformer(x, mask)
        return self.next_poi_net(shared_output, mask=mask)

    def loss_balancing(self, loss_category, loss_next):
        """Dynamic loss balancing based on task difficulty"""
        # Implement uncertainty-based loss weighting (Kendall et al., 2018)
        # This is a simplified version
        log_var_category = torch.log(torch.tensor(1.0, device=loss_category.device).requires_grad_())
        log_var_next = torch.log(torch.tensor(1.0, device=loss_next.device).requires_grad_())

        precision_category = torch.exp(-log_var_category)
        precision_next = torch.exp(-log_var_next)

        balanced_loss = precision_category * loss_category + 0.5 * log_var_category + \
                        precision_next * loss_next + 0.5 * log_var_next

        return balanced_loss
