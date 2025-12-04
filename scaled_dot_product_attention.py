import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8, dim_ff=512):
        super().__init__()
        
        # Multi-head attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Feed-forward network (two linear layers)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # --- Self-Attention Block ---
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)  # Add & Norm
        
        # --- Feed-Forward Block ---
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Add & Norm
        
        return x


# ---- Verification ----
batch_size = 32
seq_len = 10
d_model = 128

model = SimpleTransformerEncoder()
input_tensor = torch.randn(batch_size, seq_len, d_model)

output = model(input_tensor)
print("Output Shape:", output.shape)
