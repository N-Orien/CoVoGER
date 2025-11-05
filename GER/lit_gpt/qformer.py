import torch
import torch.nn as nn
import math

class AttentionLayer(nn.Module):
    """
    A generic multi-head attention layer (self-attention or cross-attention).
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.out_proj   = nn.Linear(d_model, d_model)
        self.dropout    = nn.Dropout(dropout)
        
    def forward(self, 
                query,      # (batch_size, query_len, d_model)
                key,        # (batch_size, key_len, d_model)
                value,      # (batch_size, key_len, d_model)
                mask=None):
        B, QL, _ = query.shape
        _, KL, _ = key.shape
        
        # Project inputs
        Q = self.query_proj(query)  # (B, QL, d_model)
        K = self.key_proj(key)      # (B, KL, d_model)
        V = self.value_proj(value)  # (B, KL, d_model)
        
        # Reshape for multi-head attention: (B, n_heads, seq_len, head_dim)
        Q = Q.view(B, QL, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, QL, head_dim)
        K = K.view(B, KL, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, KL, head_dim)
        V = V.view(B, KL, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, KL, head_dim)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, n_heads, QL, KL)
        
        if mask is not None:
            # mask should be broadcastable to (B, n_heads, QL, KL)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)  # (B, n_heads, QL, KL)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # (B, n_heads, QL, head_dim)
        
        # Recombine heads
        out = out.transpose(1, 2).contiguous()  # (B, QL, n_heads, head_dim)
        out = out.view(B, QL, self.d_model)     # (B, QL, d_model)
        
        return self.out_proj(out)               # (B, QL, d_model)


class TransformerFFN(nn.Module):
    """
    A simple feed-forward block used in transformer architectures:
      FFN(x) = Dropout(GELU(Linear(x))) + ...
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class QFormerLayer(nn.Module):
    """
    One layer of the Q-Former:
      1) Self-Attention among queries
      2) Cross-Attention from queries to image features
      3) Feed Forward
    """
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = AttentionLayer(d_model, n_heads, dropout)
        
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = AttentionLayer(d_model, n_heads, dropout)
        
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = TransformerFFN(d_model, hidden_dim, dropout)
        
    def forward(self, query_tokens, image_feats):
        # ----- 1) Self-Attention -----
        residual = query_tokens
        query_tokens = self.self_attn_norm(query_tokens)
        query_tokens = self.self_attn(query_tokens, query_tokens, query_tokens)
        query_tokens = residual + query_tokens
        
        # ----- 2) Cross-Attention (queries -> image_feats) -----
        residual = query_tokens
        query_tokens = self.cross_attn_norm(query_tokens)
        query_tokens = self.cross_attn(query_tokens, image_feats, image_feats)
        query_tokens = residual + query_tokens
        
        # ----- 3) Feed-Forward -----
        residual = query_tokens
        query_tokens = self.ffn_norm(query_tokens)
        query_tokens = self.ffn(query_tokens)
        query_tokens = residual + query_tokens
        
        return query_tokens


class QFormer(nn.Module):
    """
    Q-Former module with:
      - Learnable query embeddings
      - Stacked QFormerLayers
    """
    def __init__(self, 
                 num_queries=40,     # How many queries we want
                 d_model=512,        # Dimensionality for queries/features
                 n_heads=8,         
                 hidden_dim=2048,    # Hidden dim inside FFN
                 input_dim=1024,
                 output_dim=4096,
                 num_layers=2,       # Number of Q-Former layers
                 dropout=0.1):
        super().__init__()
        
        # The learnable query embeddings
#        self.query_embeddings = nn.Parameter(torch.zeros(num_queries, d_model))
        self.num_queries = num_queries
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, d_model))

        self.input_linear = nn.Linear(in_features=input_dim, out_features=d_model)
        
        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, n_heads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(in_features=d_model, out_features=output_dim)

    def forward(self, image_feats):
        """
        Args:
            image_feats: (batch_size, seq_len, d_model)
                         features from a frozen image encoder
        Returns:
            query_tokens: (batch_size, num_queries, d_model)
        """
        B = image_feats.size(0)
        # Expand the learnable queries to batch size
        # shape -> (batch_size, num_queries, d_model)
        query_tokens = self.query_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        image_feats = self.input_linear(image_feats)

        # Pass through multiple Q-Former layers
        for layer in self.layers:
            query_tokens = layer(query_tokens, image_feats)
        
        query_tokens = self.output_linear(query_tokens)

        # Output tokens can be passed to a Language Model
        return query_tokens
