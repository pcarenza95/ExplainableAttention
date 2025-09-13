import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import math
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import random
import os
import tempfile
import zipfile
import io
import lime
import lime.lime_tabular
import shap
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance

# Set page config
st.set_page_config(
    page_title="Transformer Algorithmic Task Evaluation with xAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display app title and description
st.title("🧠 Transformer Algorithmic Task Evaluation with Explainable AI (xAI)")
st.markdown("""
### Comprehensive Evaluation of Transformer Architectures with Model Interpretability

This Streamlit application evaluates different Transformer architectures on algorithmic tasks 
like sorting and addition, with a focus on length generalization capabilities and model interpretability.
""")

# Create expandable section for detailed information
with st.expander("📖 Click here for detailed information about this app", expanded=True):
    st.markdown("""
    #### MAIN FEATURES:
    1. **Multiple Transformer variants**: Baseline, RoPE, ALiBi, and Longformer
    2. **Two algorithmic tasks**: Sorting and Addition
    3. **Comprehensive performance metrics**: Accuracy, Precision, Recall, F1-score
    4. **Length generalization testing**: Evaluate how models perform on longer sequences
    5. **Explainable AI (xAI)**: SHAP and LIME for model interpretability
    6. **Attention visualization**: See what the model focuses on
    7. **Comparative analysis**: Compare different architectures
    8. **Downloadable results**: Export all results and visualizations

    #### HOW TO USE:
    1. Configure experiment parameters in the sidebar
    2. Select the algorithmic task (sorting or addition)
    3. Choose which models to train and compare
    4. Adjust model architecture parameters (dimension, layers, heads)
    5. Enable/disable xAI explanations (SHAP and LIME)
    6. Click "Run Experiments" to start training and evaluation
    7. View results in the tabbed interface
    8. Download results and visualizations using the buttons provided

    #### EXPLAINABLE AI FEATURES:
    - **SHAP (SHapley Additive exPlanations)**: Shows feature importance values
    - **LIME (Local Interpretable Model-agnostic Explanations)**: Provides local explanations
    - **Attention visualization**: Shows how the model attends to different parts of the input

    #### OUTPUTS:
    1. Training metrics and loss curves
    2. Validation performance across epochs
    3. Length generalization results
    4. Attention heatmaps for each layer and head
    5. xAI explanations (SHAP and LIME)
    6. Comparative analysis across models
    7. Downloadable CSV with all results
    8. Downloadable ZIP file with all visualizations

    #### ADDITIONAL RESOURCES:
    - [Original Transformer Paper (Attention Is All You Need)](https://arxiv.org/abs/1706.03762)
    - [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
    - [ALiBi: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
    - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
    - [SHAP: A Unified Approach to Explain Model Outputs](https://arxiv.org/abs/1705.07874)
    - [LIME: Explaining Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
    - [My GitHub](https://github.com/pcarenza95/ML_portfolio/AttentionExplain)
    

    #### NOTE: 
    Training Transformers can be computationally intensive. For faster results, 
    consider using smaller models or fewer epochs when running on CPU.
    """)

# Add a separator
st.markdown("---")

# Set device configuration
device = torch.device('cpu')
st.sidebar.write(f"Using device: {device}")

# --- Utility Functions ---
def make_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Positional Embeddings and Attention Variants ---
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings as in the original Transformer paper"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        i = torch.arange(self.dim, dtype=torch.float32, device=device)
        freqs = 1 / (10000 ** (2 * (i // 2) / self.dim))
        enc = pos * freqs
        enc[:, 0::2] = torch.sin(enc[:, 0::2])
        enc[:, 1::2] = torch.cos(enc[:, 1::2])
        return enc.unsqueeze(0)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) - https://arxiv.org/abs/2104.09864"""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]

def apply_rotary_pos_emb(x, freqs):
    """Apply rotary positional embedding to input tensor"""
    # x shape: [batch_size, num_heads, seq_len, head_dim]
    # freqs shape: [1, 1, seq_len, dim]
    
    # Extract the sequence length from x
    seq_len = x.shape[-2]
    
    # Ensure freqs matches the sequence length of x
    freqs = freqs[:, :, :seq_len, :]
    
    # Make sure freqs has the same head dimension as x
    # The rotary embedding should be applied to each head separately
    # We need to ensure the dimension matches
    head_dim = x.shape[-1]
    if freqs.shape[-1] != head_dim:
        freqs = freqs[..., :head_dim]
    
    # Reshape for rotary application - handle both even and odd dimensions
    x_rot = x.float()
    x1 = x_rot[..., 0::2]  # Even indices
    x2 = x_rot[..., 1::2]  # Odd indices
    
    # Extract cosine and sine components for the relevant dimensions
    cos = torch.cos(freqs)[..., 0::2]  # Take even indices for cos
    sin = torch.sin(freqs)[..., 0::2]  # Take even indices for sin
    
    # Make sure dimensions match
    if cos.shape[-1] != x1.shape[-1]:
        # Truncate if necessary
        min_dim = min(cos.shape[-1], x1.shape[-1])
        cos = cos[..., :min_dim]
        sin = sin[..., :min_dim]
        x1 = x1[..., :min_dim]
        x2 = x2[..., :min_dim]
    
    # Apply rotary transformation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x2 * cos + x1 * sin
    
    # Interleave the results back
    rotated_x = torch.zeros_like(x_rot)
    rotated_x[..., 0::2] = rotated_x1
    rotated_x[..., 1::2] = rotated_x2
    
    return rotated_x.type_as(x)

class ALiBiBias(nn.Module):
    """ALiBi (Attention with Linear Biases) - https://arxiv.org/abs/2108.12409"""
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.tensor(self._get_slopes(heads))
        self.register_buffer("slopes", slopes.view(1, heads, 1, 1))

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            return [start*(start**i) for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                    self._get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

    def forward(self, seq_len, device):
        # Create ALiBi bias matrix with proper dimensions [1, heads, seq_len, seq_len]
        bias = torch.arange(seq_len, device=device).float()
        bias = bias - torch.arange(seq_len, device=device).unsqueeze(1)
        bias = torch.abs(bias)  # Distance matrix
        bias = bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        bias = bias * self.slopes  # Apply slopes per head
        return -bias  # Negative because we want to bias against distant tokens

# Multihead with support for ALiBi, RoPE, recurrence, and sparse (Longformer/Performer)
class MultiHeadAttention(nn.Module):
    """Multi-head attention with various positional encoding and attention variants"""
    def __init__(self, dim, heads=4, pos_mode='learned', attn_type='full', memory_len=0, window_size=5):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.pos_mode = pos_mode
        self.attn_type = attn_type
        self.memory_len = memory_len
        self.window_size = window_size
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Initialize positional embeddings based on mode
        if pos_mode == 'learned':
            self.pos_emb = nn.Embedding(2048, dim)
        elif pos_mode == 'sinusoidal':
            self.pos_emb = SinusoidalPosEmb(dim)
        elif pos_mode == 'rope':
            self.rope = RotaryPositionalEmbedding(self.head_dim)
        
        # Initialize attention bias if using ALiBi
        if attn_type == 'alibi':
            self.alibi = ALiBiBias(heads)
        else:
            self.alibi = None

        # Store window size for Longformer
        self.window_size = window_size

    def forward(self, x, mask=None, memory=None):
        B, T, C = x.shape
        H = self.heads
        
        # Apply query, key, value projections
        qkv = self.to_qkv(x).reshape(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Apply positional encodings
        if self.pos_mode == 'rope':
            freqs = self.rope(T, x.device)
            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)
        elif self.pos_mode == 'learned':
            # For learned positional embeddings, we need to create position indices
            pos_indices = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            pos_emb = self.pos_emb(pos_indices).reshape(1, T, H, C // H).permute(0, 2, 1, 3)
            q = q + pos_emb
            k = k + pos_emb
        elif self.pos_mode == 'sinusoidal':
            # For sinusoidal, use the SinusoidalPosEmb class
            pos_emb = self.pos_emb(T, x.device).reshape(1, T, H, C // H).permute(0, 2, 1, 3)
            q = q + pos_emb
            k = k + pos_emb

        # Handle memory (for recurrent transformers)
        if memory is not None:
            k_mem, v_mem = memory
            k = torch.cat([k_mem, k], dim=2)
            v = torch.cat([v_mem, v], dim=2)
            memLen = k_mem.shape[2]
        else:
            memLen = 0

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.attn_type == 'longformer':
            seq_len = attn_scores.size(-1)
            
            # Create band mask efficiently
            band_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            
            # Create band mask using broadcasting
            i = torch.arange(seq_len, device=x.device)
            j = torch.arange(seq_len, device=x.device)
            
            # Create mask for positions within window
            distance = torch.abs(i[:, None] - j[None, :])
            band_mask = distance <= (self.window_size // 2)
            
            # Convert to attention mask (0 for allowed, -inf for masked)
            band_mask = band_mask.float()
            band_mask = (1.0 - band_mask) * -1e9  # 0 for allowed, -inf for masked
            
            attn_scores = attn_scores + band_mask.unsqueeze(0).unsqueeze(0)

        # Apply ALiBi bias
        elif self.attn_type == 'alibi' and self.alibi is not None:
            # ALiBi bias should be applied to the attention scores
            # The bias matrix should have shape [1, heads, T, T + memLen]
            alibi_bias = self.alibi(T + memLen, x.device)
            attn_scores = attn_scores + alibi_bias

        
        # Apply input mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention dropout during training
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)

        # Compute output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        
        # Return output and memory if needed
        if memory is not None:
            new_memory = (k[:, :, -self.memory_len:], v[:, :, -self.memory_len:])
            return self.to_out(out), new_memory, attn_weights
        
        return self.to_out(out), attn_weights

# Transformer Block
class TransformerBlock(nn.Module):
    """Standard transformer block with pre-normalization"""
    def __init__(self, dim, heads=4, ff_mult=4, pos_mode='learned', 
                 attn_type='full', memory_len=0, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, pos_mode, attn_type, memory_len)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, memory=None):
        # Self-attention with residual connection
        if memory is not None:
            attn_out, new_memory, attn_weights = self.attn(self.norm1(x), mask, memory)
            x = x + self.dropout(attn_out)
        else:
            attn_out, attn_weights = self.attn(self.norm1(x), mask)
            x = x + self.dropout(attn_out)
            new_memory = None
        
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        
        return x, new_memory, attn_weights

# Transformer Model with optional recurrence/memory
class TransformerModel(nn.Module):
    """Transformer model with configurable architecture and attention mechanisms"""
    def __init__(self, dim=64, depth=4, heads=4, pos_mode='learned', 
                 attn_type='full', memory_len=0, vocab_size=10, ff_mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.memory_len = memory_len
        self.vocab_size = vocab_size
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_mult, pos_mode, attn_type, memory_len, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None, memory=None):
        # Token embeddings
        x = self.token_emb(x)
        x = self.dropout(x)
        
        # Initialize memory if not provided
        if memory is None and self.memory_len > 0:
            B, T = x.shape[0], x.shape[1]
            memory = [
                torch.zeros(B, self.blocks[0].attn.heads, self.memory_len, self.dim // self.blocks[0].attn.heads, 
                           device=x.device) for _ in range(self.depth)
            ]
        
        # Process through transformer blocks
        new_memory = [] if self.memory_len > 0 else None
        attention_weights = []
        for i, blk in enumerate(self.blocks):
            if memory is not None:
                x, mem, attn = blk(x, mask, memory[i] if memory is not None else None)
                if new_memory is not None:
                    new_memory.append(mem)
                attention_weights.append(attn)
            else:
                x, _, attn = blk(x, mask)
                attention_weights.append(attn)
        
        # Final normalization and output
        x = self.norm(x)
        logits = self.to_logits(x)
        
        return logits, new_memory, attention_weights

# --- Algorithmic Task Datasets ---
class AlgorithmicTaskDataset:
    def __init__(self, task='sorting', seq_length=10, vocab_size=10, num_samples=10000):
        self.task = task
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        
    def generate_sorting_data(self):
        data = []
        targets = []
        
        for _ in range(self.num_samples):
            # Generate random sequence
            seq = np.random.randint(1, self.vocab_size, self.seq_length)
            
            # Target is the sorted sequence
            target = np.sort(seq)
            
            data.append(seq)
            targets.append(target)
            
        return np.array(data), np.array(targets)
    
    def generate_addition_data(self):
        data = []
        targets = []
        
        for _ in range(self.num_samples):
            # Generate two numbers to add
            num1 = np.random.randint(0, self.vocab_size // 2, self.seq_length // 2)
            num2 = np.random.randint(0, self.vocab_size // 2, self.seq_length // 2)
            
            # Create input sequence: [num1, +, num2]
            seq = np.concatenate([num1, [self.vocab_size - 1], num2])
            
            # Calculate target (sum)
            total = np.sum(num1) + np.sum(num2)
            # Represent as a sequence of digits
            target_str = str(total)
            target = np.array([int(d) for d in target_str] + [0] * (self.seq_length - len(target_str)))
            
            data.append(seq)
            targets.append(target)
            
        return np.array(data), np.array(targets)
    
    def generate_data(self):
        if self.task == 'sorting':
            return self.generate_sorting_data()
        elif self.task == 'addition':
            return self.generate_addition_data()
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def get_dataset(self):
        data, targets = self.generate_data()
        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def get_examples(self, num_examples=5):
        """Get examples from the dataset with explanations"""
        examples = []
        data, targets = self.generate_data()
        
        for i in range(min(num_examples, len(data))):
            if self.task == 'sorting':
                explanation = f"Input: {data[i]} → Sorted: {targets[i]}"
            elif self.task == 'addition':
                # For addition, just show the input and target without detailed explanation
                explanation = f"Input: {data[i]} → Target: {targets[i]}"
            
            examples.append({
                'input': data[i],
                'target': targets[i],
                'explanation': explanation
            })
        
        return examples

# --- Training and Evaluation Functions ---
def train_model(model, train_data, train_targets, val_data, val_targets, 
                epochs=10, batch_size=32, lr=0.001, device='cpu', warmup_epochs=3, progress_bar=None, status_text=None, model_name=""):
    """Train the transformer model on algorithmic tasks"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    grad_norms = []
    
    start_time = time.time()
    
    # Warm-up scheduler
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_grad_norms = []
        
        # Use tqdm for progress bar if no Streamlit progress bar is provided
        if progress_bar is None:
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            batch_iter = train_loader
        
        for batch_idx, (data, target) in enumerate(batch_iter):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _, _ = model(data)
            
            # Handle sequence length mismatch - truncate to the minimum length
            seq_len = min(output.size(1), target.size(1))
            output = output[:, :seq_len, :]  # [batch_size, seq_len, vocab_size]
            target = target[:, :seq_len]     # [batch_size, seq_len]
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            target = target.reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            
            # Track gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar if provided
            if progress_bar is not None and status_text is not None:
                progress = (epoch * len(train_loader) + batch_idx + 1) / (epochs * len(train_loader))
                progress_bar.progress(progress)
                status_text.text(f"{model_name} - Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        train_losses.append(avg_train_loss)
        grad_norms.append(avg_grad_norm)
        
        # Update learning rate (warmup for first few epochs)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _, _ = model(data)
                
                # Handle sequence length mismatch
                seq_len = min(output.size(1), target.size(1))
                output = output[:, :seq_len, :]
                target = target[:, :seq_len]
                
                # Reshape for loss calculation
                output_flat = output.reshape(-1, output.size(-1))
                target_flat = target.reshape(-1)
                
                loss = criterion(output_flat, target_flat)
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(output, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate additional metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        
        # Update learning rate based on validation loss
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)
        
        # Update status text
        if status_text is not None:
            status_text.text(f"{model_name} - Epoch {epoch+1}/{epochs} completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.4f}")
    
    training_time = time.time() - start_time
    
    return train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s, grad_norms, training_time

def test_length_generalization(model, original_length, new_lengths, task='sorting', 
                              vocab_size=10, device='cpu'):
    """Test model generalization to longer sequences"""
    results = {}
    
    for length in new_lengths:
        # Generate test data with new length
        if task == 'sorting':
            test_data = np.random.randint(1, vocab_size, (100, length))
            test_targets = np.sort(test_data, axis=1)
        elif task == 'addition':
            test_data = []
            test_targets = []
            for _ in range(100):
                num1 = np.random.randint(0, vocab_size // 2, length // 2)
                num2 = np.random.randint(0, vocab_size // 2, length // 2)
                seq = np.concatenate([num1, [vocab_size - 1], num2])
                total = np.sum(num1) + np.sum(num2)
                target_str = str(total)
                # Make target the same length as input for consistency
                target = np.array([int(d) for d in target_str] + [0] * (length - len(target_str)))
                test_data.append(seq)
                test_targets.append(target)
            test_data = np.array(test_data)
            test_targets = np.array(test_targets)
        
        test_data = torch.tensor(test_data, dtype=torch.long).to(device)
        test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)
        
        # Test the model
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size]
                batch_targets = test_targets[i:i+batch_size]
                
                output, _, _ = model(batch_data)
                
                # Handle sequence length mismatch
                seq_len = min(output.size(1), batch_targets.size(1))
                output = output[:, :seq_len, :]
                batch_targets = batch_targets[:, :seq_len]
                
                predictions = torch.argmax(output, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_targets.cpu().numpy().flatten())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        results[length] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results

def visualize_attention(model, data_sample, task='sorting', device='cpu'):
    """Visualize attention patterns for a sample"""
    model.eval()
    with torch.no_grad():
        output, _, attention_weights = model(data_sample.unsqueeze(0).to(device))
    
    # Plot attention weights for each layer and head
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)  # [batch_size, num_heads, seq_len, seq_len]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4*num_heads, 4*num_layers))
    
    if num_layers == 1:
        axes = [axes]
    if num_heads == 1:
        for i in range(num_layers):
            axes[i] = [axes[i]]
    
    for layer in range(num_layers):
        for head in range(num_heads):
            attn_map = attention_weights[layer][0, head].cpu().numpy()
            ax = axes[layer][head]
            sns.heatmap(attn_map, ax=ax, cmap='viridis')
            ax.set_title(f'Layer {layer+1}, Head {head+1}')
    
    plt.tight_layout()
    return fig, attention_weights


def explain_with_shap(model, data_sample, task='sorting', device='cpu'):
    """Explain model predictions using SHAP"""
    model.eval()
    
    # Get the sequence length from the sample
    seq_len = data_sample.shape[0]
    
    # Create a function that returns model outputs for the first position only
    def model_fn(x):
        # Convert numpy array to tensor
        x_tensor = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            output, _, _ = model(x_tensor)
            # Return probabilities for the first token prediction only
            # This gives us [batch_size, vocab_size] which SHAP expects
            return F.softmax(output[:, 0, :], dim=-1).cpu().numpy()
    
    # Create background data with the same sequence length as the sample
    background_data = np.random.randint(1, 10, (50, seq_len))
    
    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(model_fn, background_data)
    
    # Calculate SHAP values for the sample
    sample_reshaped = data_sample.numpy().reshape(1, -1)
    
    # Get SHAP values for all classes
    shap_values = explainer.shap_values(sample_reshaped)
    
    # Plot SHAP values for the predicted class
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # If we have multiple classes, shap_values will be a list
    if isinstance(shap_values, list):
        # Get the predicted class
        with torch.no_grad():
            output, _, _ = model(data_sample.unsqueeze(0).to(device))
            predicted_class = torch.argmax(output[0, 0, :]).item()

        # Use the SHAP values for the predicted class
        shap_to_plot = shap_values[predicted_class]
    else:
        shap_to_plot = shap_values
    
    # Create a bar plot of SHAP values
    feature_names = [f'Pos {i+1}' for i in range(seq_len)]
    shap.summary_plot(shap_to_plot, sample_reshaped, feature_names=feature_names, 
                     plot_type="bar", show=False)
    
    plt.title(f'SHAP Feature Importance for {task.capitalize()} Task')
    plt.tight_layout()
    
    return fig, shap_values

def explain_with_lime(model, data_sample, task='sorting', device='cpu'):
    """Explain model predictions using LIME"""
    model.eval()
    
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.random.randint(1, 10, (100, data_sample.shape[0])),
        feature_names=[f"Pos_{i}" for i in range(data_sample.shape[0])],
        mode="classification",
        class_names=[str(i) for i in range(10)]  # Assuming vocab_size=10
    )
    
    # Create prediction function that returns 2D output
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.long).to(device)
        with torch.no_grad():
            output, _, _ = model(x_tensor)
            # Return probabilities for the first position only
            # This converts 3D [batch, seq_len, vocab] to 2D [batch, vocab]
            return F.softmax(output[:, 0, :], dim=-1).cpu().numpy()
    
    # Explain the instance - focus on predicting the first token
    explanation = explainer.explain_instance(
        data_sample.numpy(), 
        predict_fn, 
        num_features=data_sample.shape[0],
        num_samples=100 
    )
    
    # Create plot
    fig = plt.figure(figsize=(10, 6))
    exp_list = explanation.as_list()
    features = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]
    
    colors = ['red' if w < 0 else 'green' for w in weights]
    plt.barh(features, weights, color=colors)
    plt.xlabel("Feature Importance")
    plt.title("LIME Explanation (First Token Prediction)")
    plt.tight_layout()
    
    return fig, explanation

# --- Main Application Logic ---
def main():
    # Sidebar configuration
    st.sidebar.header("🧪 Experiment Configuration")
    
    # Task selection
    task = st.sidebar.selectbox(
        "Select Algorithmic Task",
        ["sorting", "addition"],
        help="Choose the task to evaluate models on"
    )
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = {
        "Baseline Transformer": {"pos_mode": "learned", "attn_type": "full"},
        "RoPE Transformer": {"pos_mode": "rope", "attn_type": "full"},
        "ALiBi Transformer": {"pos_mode": "learned", "attn_type": "alibi"},
        "Longformer": {"pos_mode": "learned", "attn_type": "longformer"}
    }
    
    selected_models = []
    for model_name in model_options:
        if st.sidebar.checkbox(model_name, value=(model_name == "Baseline Transformer")):
            selected_models.append(model_name)
    
    # Model architecture parameters
    st.sidebar.subheader("Model Architecture")
    dim = st.sidebar.slider("Model Dimension", min_value=32, max_value=256, value=64, step=32)
    depth = st.sidebar.slider("Number of Layers", min_value=2, max_value=8, value=4, step=1)
    heads = st.sidebar.slider("Number of Heads", min_value=2, max_value=8, value=4, step=1)
    ff_mult = st.sidebar.slider("Feed-Forward Multiplier", min_value=2, max_value=8, value=4, step=1)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    seq_length = st.sidebar.slider("Sequence Length", min_value=5, max_value=20, value=10, step=1)
    vocab_size = st.sidebar.slider("Vocabulary Size", min_value=5, max_value=20, value=10, step=1)
    num_samples = st.sidebar.slider("Number of Samples", min_value=1000, max_value=10000, value=5000, step=1000)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
    epochs = st.sidebar.slider("Epochs", min_value=5, max_value=50, value=20, step=5)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=1e-4, max_value=1e-2, value=1e-3, step=1e-4, format="%.4f")
    
    # xAI options
    st.sidebar.subheader("Explainable AI Options")
    enable_shap = st.sidebar.checkbox("Enable SHAP Explanations", value=True)
    enable_lime = st.sidebar.checkbox("Enable LIME Explanations", value=True)
    enable_attention = st.sidebar.checkbox("Enable Attention Visualization", value=True)
    
    # Test options
    st.sidebar.subheader("Generalization Testing")
    test_generalization = st.sidebar.checkbox("Test Length Generalization", value=True)
    generalization_lengths = st.sidebar.multiselect(
        "Test Lengths",
        options=[5, 10, 15, 20, 25, 30],
        default=[10, 15, 20]
    )
    
    # Run button
    if st.sidebar.button("🚀 Run Experiments", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train!")
            return
        
        # Set seed for reproducibility
        # set_seed(42)
        
        # Generate dataset
        dataset = AlgorithmicTaskDataset(
            task=task,
            seq_length=seq_length,
            vocab_size=vocab_size,
            num_samples=num_samples
        )
        
        # Split data
        data, targets = dataset.get_dataset()
        split_idx = int(0.8 * len(data))
        train_data, val_data = data[:split_idx], data[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        
        # Display dataset info
        st.info(f"Generated {len(data)} samples for {task} task with sequence length {seq_length}")
        
        # Show examples
        examples = dataset.get_examples(5)
        with st.expander("📋 Dataset Examples"):
            for i, example in enumerate(examples):
                st.write(f"**Example {i+1}:** {example['explanation']}")
        
        # Initialize results storage
        results = {}
        training_times = {}
        
        # Create progress bar and status area
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train selected models
        for model_name in selected_models:
            st.subheader(f"🧠 Training {model_name}")
            
            # Get model configuration
            config = model_options[model_name]
            
            # Initialize model
            model = TransformerModel(
                dim=dim,
                depth=depth,
                heads=heads,
                pos_mode=config["pos_mode"],
                attn_type=config["attn_type"],
                vocab_size=vocab_size,
                ff_mult=ff_mult
            ).to(device)
            
            # Train model
            train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s, grad_norms, training_time = train_model(
                model=model,
                train_data=train_data,
                train_targets=train_targets,
                val_data=val_data,
                val_targets=val_targets,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                device=device,
                progress_bar=progress_bar,
                status_text=status_text,
                model_name=model_name
            )
            
            # Store results
            results[model_name] = {
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'val_precisions': val_precisions,
                'val_recalls': val_recalls,
                'val_f1s': val_f1s,
                'grad_norms': grad_norms
            }
            
            training_times[model_name] = training_time
            
            # Display training results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Training Loss", f"{train_losses[-1]:.4f}")
                st.metric("Final Validation Loss", f"{val_losses[-1]:.4f}")
            with col2:
                st.metric("Final Accuracy", f"{val_accuracies[-1]:.4f}")
                st.metric("Final F1 Score", f"{val_f1s[-1]:.4f}")
            with col3:
                st.metric("Training Time", f"{training_time:.2f}s")
                st.metric("Best Accuracy", f"{max(val_accuracies):.4f}")
        
        # Clear progress bar
        progress_bar.empty()
        status_text.empty()
        
        # --- Results Visualization ---
        st.header("📊 Results Analysis")
        
        # Create tabs for different types of results
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Training Curves", 
            "🎯 Performance Metrics", 
            "🔍 Model Interpretability",
            "📏 Length Generalization",
            "📋 Comparative Analysis"
        ])
        
        with tab1:
            st.subheader("Training and Validation Curves")
            
            # Plot training and validation loss
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Loss curves
            for model_name in selected_models:
                axes[0, 0].plot(results[model_name]['train_losses'], label=f'{model_name} Train')
                axes[0, 0].plot(results[model_name]['val_losses'], label=f'{model_name} Val', linestyle='--')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy curves
            for model_name in selected_models:
                axes[0, 1].plot(results[model_name]['val_accuracies'], label=model_name)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gradient norms
            for model_name in selected_models:
                axes[1, 0].plot(results[model_name]['grad_norms'], label=model_name)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Average Gradient Norms')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # F1 score
            for model_name in selected_models:
                axes[1, 1].plot(results[model_name]['val_f1s'], label=model_name)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Validation F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Final Performance Metrics")
            
            # Create metrics table
            metrics_data = []
            for model_name in selected_models:
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': results[model_name]['val_accuracies'][-1],
                    'Precision': results[model_name]['val_precisions'][-1],
                    'Recall': results[model_name]['val_recalls'][-1],
                    'F1 Score': results[model_name]['val_f1s'][-1],
                    'Training Time (s)': training_times[model_name]
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'Training Time (s)': '{:.2f}'
            }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']))
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(selected_models))
            width = 0.2
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            metric_keys = {
                    'Accuracy': 'val_accuracies',
                    'Precision': 'val_precisions', 
                    'Recall': 'val_recalls',
                    'F1 Score': 'val_f1s'
            }
            
            for i, metric in enumerate(metrics):
                values = [results[model_name][metric_keys[metric]][-1] for model_name in selected_models]
                ax.bar(x + i * width, values, width, label=metric, color=colors[i])
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Final Performance Metrics Comparison')
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(selected_models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Model Interpretability and Explainability")
            
            if not selected_models:
                st.warning("No models selected for interpretability analysis")
            else:
                # Select a sample for explanation (use the first validation sample)
                sample_idx = 0  # Fixed sample index as requested
                sample_data = val_data[sample_idx]
                sample_target = val_targets[sample_idx]
                
                st.write(f"**Sample Input:** {sample_data.tolist()}")
                st.write(f"**True Target:** {sample_target.tolist()}")
                
                # Get predictions for the sample
                for model_name in selected_models:
                    model = results[model_name]['model']
                    model.eval()
                    with torch.no_grad():
                        output, _, _ = model(sample_data.unsqueeze(0).to(device))
                        prediction = torch.argmax(output, dim=-1)[0]
                    
                    st.write(f"**{model_name} Prediction:** {prediction.cpu().numpy().tolist()}")
                    accuracy = accuracy_score(sample_target.numpy().flatten(), prediction.cpu().numpy().flatten())
                    st.write(f"**Sample Accuracy:** {accuracy:.4f}")
                
                # Loop through all selected models for detailed explanation
                for explain_model in selected_models:
                    model = results[explain_model]['model']
                    st.markdown(f"### Explanations for **{explain_model}**")
                
                    # Create columns for different explanation methods
                    col1, col2 = st.columns(2)
                
                    with col1:
                        if enable_attention:
                            st.subheader("Attention Visualization")
                            try:
                                attn_fig, attn_weights = visualize_attention(model, sample_data, task, device)
                                st.pyplot(attn_fig)
                                st.caption(f"Attention patterns for {explain_model}")
                            except Exception as e:
                                st.error(f"Error in attention visualization for {explain_model}: {e}")
                
                    with col2:
                        if enable_shap:
                            st.subheader("SHAP Explanation")
                            try:
                                shap_fig, shap_values = explain_with_shap(model, sample_data, task, device)
                                st.pyplot(shap_fig)
                                st.caption(f"SHAP explanation for {explain_model}")
                            except Exception as e:
                                st.error(f"Error in SHAP explanation for {explain_model}: {e}")
                
                    st.markdown("---")  # separator between models
                    if enable_lime:
                        st.subheader("LIME Explanation")
                        try:
                            lime_fig, lime_exp = explain_with_lime(model, sample_data, task, device)
                            st.pyplot(lime_fig)
                            if lime_exp is not None:
                                st.caption(f"LIME explanation for {explain_model}")
                        except Exception as e:
                            st.error(f"Error in LIME explanation: {e}")             
            
        
        with tab4:
            st.subheader("Length Generalization Analysis")
            
            if test_generalization and generalization_lengths:
                generalization_results = {}
                
                for model_name in selected_models:
                    model = results[model_name]['model']
                    gen_results = test_length_generalization(
                        model=model,
                        original_length=seq_length,
                        new_lengths=generalization_lengths,
                        task=task,
                        vocab_size=vocab_size,
                        device=device
                    )
                    generalization_results[model_name] = gen_results
                
                # Display results in a table
                gen_data = []
                for model_name in selected_models:
                    for length in generalization_lengths:
                        gen_data.append({
                            'Model': model_name,
                            'Length': length,
                            'Accuracy': generalization_results[model_name][length]['accuracy'],
                            'F1 Score': generalization_results[model_name][length]['f1']
                        })
                
                gen_df = pd.DataFrame(gen_data)
                st.dataframe(gen_df.style.format({
                    'Accuracy': '{:.4f}',
                    'F1 Score': '{:.4f}'
                }).highlight_max(subset=['Accuracy', 'F1 Score']))
                
                # Plot generalization results
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                for model_name in selected_models:
                    accuracies = [generalization_results[model_name][length]['accuracy'] for length in generalization_lengths]
                    axes[0].plot(generalization_lengths, accuracies, 'o-', label=model_name)
                
                axes[0].set_xlabel('Sequence Length')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_title('Length Generalization - Accuracy')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                for model_name in selected_models:
                    f1_scores = [generalization_results[model_name][length]['f1'] for length in generalization_lengths]
                    axes[1].plot(generalization_lengths, f1_scores, 'o-', label=model_name)
                
                axes[1].set_xlabel('Sequence Length')
                axes[1].set_ylabel('F1 Score')
                axes[1].set_title('Length Generalization - F1 Score')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Enable length generalization testing in the sidebar to see results here.")
        
        with tab5:
            st.subheader("Comparative Analysis")
            
            # Create comprehensive comparison
            comparison_data = []
            for model_name in selected_models:
                # Calculate learning metrics
                final_acc = results[model_name]['val_accuracies'][-1]
                best_acc = max(results[model_name]['val_accuracies'])
                
                # Fix the accuracy gain calculation
                if len(results[model_name]['val_accuracies']) > 0:
                    acc_gain = max(results[model_name]['val_accuracies']) - results[model_name]['val_accuracies'][0]
                else:
                    acc_gain = 0
                
                convergence_epoch = np.argmax(results[model_name]['val_accuracies']) + 1
                
                comparison_data.append({
                    'Model': model_name,
                    'Final Accuracy': final_acc,
                    'Best Accuracy': best_acc,
                    'Accuracy Gain': acc_gain,
                    'Convergence Epoch': convergence_epoch,
                    'Training Time (s)': training_times[model_name],
                    'Efficiency (Acc/s)': best_acc / training_times[model_name] if training_times[model_name] > 0 else 0
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.format({
                'Final Accuracy': '{:.4f}',
                'Best Accuracy': '{:.4f}',
                'Accuracy Gain': '{:.4f}',
                'Efficiency (Acc/s)': '{:.6f}'
            }).highlight_max(subset=['Final Accuracy', 'Best Accuracy', 'Accuracy Gain', 'Efficiency (Acc/s)'])
            .highlight_min(subset=['Convergence Epoch', 'Training Time (s)']))
            
            # Radar chart for comprehensive comparison
            metrics = ['Final Accuracy', 'Best Accuracy', 'Accuracy Gain', 'Efficiency (Acc/s)']
            metrics_normalized = []
            
            for metric in metrics:
                values = comparison_df[metric].values
                # Normalize to 0-1 scale
                if max(values) > min(values):
                    normalized = (values - min(values)) / (max(values) - min(values))
                else:
                    normalized = np.ones_like(values)
                metrics_normalized.append(normalized)
            
            metrics_normalized = np.array(metrics_normalized)
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the circle
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(selected_models)))
            
            for i, model_name in enumerate(selected_models):
                values = metrics_normalized[:, i].tolist()
                values += values[:1]  # Close the circle
                ax.plot(angles, values, 'o-', color=colors[i], label=model_name)
                ax.fill(angles, values, color=colors[i], alpha=0.1)
            
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
            ax.set_title('Model Comparison Radar Chart', size=14, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            st.pyplot(fig)
        
        # --- Download Results ---
        st.header("💾 Download Results")
        
        # Create downloadable results
        results_dict = {
            'task': task,
            'sequence_length': seq_length,
            'vocab_size': vocab_size,
            'models_trained': selected_models,
            'training_parameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'results': {model: {k: v for k, v in results[model].items() if k != 'model'} for model in results},
            'training_times': training_times,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON string
        results_json = json.dumps(results_dict, indent=2, default=str)
        
        # Create download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Results JSON",
                data=results_json,
                file_name=f"transformer_results_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create a zip file with all visualizations
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                    # Add JSON results
                    zipf.writestr('results.json', results_json)
                    
                    # Save all visualizations to the zip file
                    visualization_count = 0
                    
                    # 1. Save training curves
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    for model_name in selected_models:
                        axes[0, 0].plot(results[model_name]['train_losses'], label=f'{model_name} Train')
                        axes[0, 0].plot(results[model_name]['val_losses'], label=f'{model_name} Val', linestyle='--')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].set_title('Training and Validation Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    for model_name in selected_models:
                        axes[0, 1].plot(results[model_name]['val_accuracies'], label=model_name)
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].set_title('Validation Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    for model_name in selected_models:
                        axes[1, 0].plot(results[model_name]['grad_norms'], label=model_name)
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Gradient Norm')
                    axes[1, 0].set_title('Average Gradient Norms')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    for model_name in selected_models:
                        axes[1, 1].plot(results[model_name]['val_f1s'], label=model_name)
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('F1 Score')
                    axes[1, 1].set_title('Validation F1 Score')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    zipf.writestr('training_curves.png', buf.read())
                    buf.close()
                    visualization_count += 1
                    plt.close(fig)
                    
                    # 2. Save performance metrics comparison
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(selected_models))
                    width = 0.2
                    
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
                    metric_keys = {
                        'Accuracy': 'val_accuracies',
                        'Precision': 'val_precisions', 
                        'Recall': 'val_recalls',
                        'F1 Score': 'val_f1s'
                    }
                    
                    for i, metric in enumerate(metrics):
                        values = [results[model_name][metric_keys[metric]][-1] for model_name in selected_models]
                        ax.bar(x + i * width, values, width, label=metric, color=colors[i])
                    
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Score')
                    ax.set_title('Final Performance Metrics Comparison')
                    ax.set_xticks(x + 1.5 * width)
                    ax.set_xticklabels(selected_models, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    zipf.writestr('performance_comparison.png', buf.read())
                    buf.close()
                    visualization_count += 1
                    plt.close(fig)
                    
                    # 3. Save attention visualizations for each model
                    sample_data = val_data[0]  # Use first validation sample
                    for model_name in selected_models:
                        try:
                            model = results[model_name]['model']
                            attn_fig, attn_weights = visualize_attention(model, sample_data, task, device)
                            buf = io.BytesIO()
                            attn_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            zipf.writestr(f'attention_{model_name.replace(" ", "_")}.png', buf.read())
                            buf.close()
                            visualization_count += 1
                            plt.close(attn_fig)
                        except Exception as e:
                            st.warning(f"Could not save attention visualization for {model_name}: {e}")
                    
                    # 4. Save SHAP explanations for each model
                    for model_name in selected_models:
                        try:
                            model = results[model_name]['model']
                            shap_fig, shap_values = explain_with_shap(model, sample_data, task, device)
                            buf = io.BytesIO()
                            shap_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            zipf.writestr(f'shap_{model_name.replace(" ", "_")}.png', buf.read())
                            buf.close()
                            visualization_count += 1
                            plt.close(shap_fig)
                        except Exception as e:
                            st.warning(f"Could not save SHAP explanation for {model_name}: {e}")
                    
                    # 5. Save LIME explanations for each model
                    for model_name in selected_models:
                        try:
                            model = results[model_name]['model']
                            lime_fig, lime_exp = explain_with_lime(model, sample_data, task, device)
                            buf = io.BytesIO()
                            lime_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            zipf.writestr(f'lime_{model_name.replace(" ", "_")}.png', buf.read())
                            buf.close()
                            visualization_count += 1
                            plt.close(lime_fig)
                        except Exception as e:
                            st.warning(f"Could not save LIME explanation for {model_name}: {e}")
                    
                    # 6. Save length generalization plots if enabled
                    if test_generalization and generalization_lengths:
                        generalization_results = {}
                        
                        for model_name in selected_models:
                            model = results[model_name]['model']
                            gen_results = test_length_generalization(
                                model=model,
                                original_length=seq_length,
                                new_lengths=generalization_lengths,
                                task=task,
                                vocab_size=vocab_size,
                                device=device
                            )
                            generalization_results[model_name] = gen_results
                        
                        # Plot generalization results
                        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                        
                        for model_name in selected_models:
                            accuracies = [generalization_results[model_name][length]['accuracy'] for length in generalization_lengths]
                            axes[0].plot(generalization_lengths, accuracies, 'o-', label=model_name)
                        
                        axes[0].set_xlabel('Sequence Length')
                        axes[0].set_ylabel('Accuracy')
                        axes[0].set_title('Length Generalization - Accuracy')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        for model_name in selected_models:
                            f1_scores = [generalization_results[model_name][length]['f1'] for length in generalization_lengths]
                            axes[1].plot(generalization_lengths, f1_scores, 'o-', label=model_name)
                        
                        axes[1].set_xlabel('Sequence Length')
                        axes[1].set_ylabel('F1 Score')
                        axes[1].set_title('Length Generalization - F1 Score')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        zipf.writestr('length_generalization.png', buf.read())
                        buf.close()
                        visualization_count += 1
                        plt.close(fig)
                    
                    # 7. Save radar chart comparison
                    comparison_data = []
                    for model_name in selected_models:
                        final_acc = results[model_name]['val_accuracies'][-1]
                        best_acc = max(results[model_name]['val_accuracies'])
                        
                        if len(results[model_name]['val_accuracies']) > 0:
                            acc_gain = max(results[model_name]['val_accuracies']) - results[model_name]['val_accuracies'][0]
                        else:
                            acc_gain = 0
                        
                        comparison_data.append({
                            'Model': model_name,
                            'Final Accuracy': final_acc,
                            'Best Accuracy': best_acc,
                            'Accuracy Gain': acc_gain,
                            'Training Time (s)': training_times[model_name],
                            'Efficiency (Acc/s)': best_acc / training_times[model_name] if training_times[model_name] > 0 else 0
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    metrics = ['Final Accuracy', 'Best Accuracy', 'Accuracy Gain', 'Efficiency (Acc/s)']
                    metrics_normalized = []
                    
                    for metric in metrics:
                        values = comparison_df[metric].values
                        if max(values) > min(values):
                            normalized = (values - min(values)) / (max(values) - min(values))
                        else:
                            normalized = np.ones_like(values)
                        metrics_normalized.append(normalized)
                    
                    metrics_normalized = np.array(metrics_normalized)
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                    angles += angles[:1]
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_models)))
                    
                    for i, model_name in enumerate(selected_models):
                        values = metrics_normalized[:, i].tolist()
                        values += values[:1]
                        ax.plot(angles, values, 'o-', color=colors[i], label=model_name)
                        ax.fill(angles, values, color=colors[i], alpha=0.1)
                    
                    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
                    ax.set_title('Model Comparison Radar Chart', size=14, pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    zipf.writestr('radar_comparison.png', buf.read())
                    buf.close()
                    visualization_count += 1
                    plt.close(fig)
                    
                    # Add a summary file
                    summary = f"""
                    TRANSFORMER EXPERIMENT RESULTS SUMMARY
                    ======================================
                    
                    Task: {task}
                    Sequence Length: {seq_length}
                    Vocabulary Size: {vocab_size}
                    Models Trained: {', '.join(selected_models)}
                    Training Epochs: {epochs}
                    Batch Size: {batch_size}
                    Learning Rate: {learning_rate}
                    
                    Total Visualizations Saved: {visualization_count}
                    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Files included:
                    - results.json: Complete experiment results
                    - training_curves.png: Training and validation curves
                    - performance_comparison.png: Final performance metrics
                    - attention_*.png: Attention visualizations for each model
                    - shap_*.png: SHAP explanations for each model
                    - lime_*.png: LIME explanations for each model
                    - length_generalization.png: Length generalization results (if enabled)
                    - radar_comparison.png: Radar chart comparison
                    """
                    zipf.writestr('SUMMARY.txt', summary)
                
                # Read the zip file for download
                with open(tmp_file.name, 'rb') as f:
                    zip_data = f.read()
                
                st.download_button(
                    label=f"📦 Download All Visualizations ({visualization_count} files)",
                    data=zip_data,
                    file_name=f"transformer_visualizations_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        st.success("✅ Experiment completed successfully!")

# Run the app
if __name__ == "__main__":
    main()