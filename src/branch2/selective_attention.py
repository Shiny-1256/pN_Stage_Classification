"""
Selective Neighborhood Attention (SNA) Module for Branch 2.
Implements Equations (1)-(6) from Tauqeer et al. (Scientific Reports 2025):
- Gated attention score e_0,(i,j) with learned sigmoid gating g(F_0,0, F_i,j)
- Softmax normalized attention coefficients alpha_0,(i,j) across 8 neighbors
- Dynamic Top-K (K=4) neighbor selection to protect tumor-normal boundaries
- Scaled neighbor features H_k = alpha_k * F_k
- Context combination X = Concat(F_0,0, H_1, ..., H_4) (shape: 5 x 1024)
- Contextual Self-Attention block -> R = sigma(Wx * beta * X)
- Multi-layer patch classification head: MLP -> BN -> ReLU -> Pool -> MLP -> Softmax
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveNeighborhoodAttention(nn.Module):
    """
    Selective Neighborhood Attention mechanism with Top-K neighbor gating
    and contextual self-attention for patch-level tumor classification.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        attn_dim: int = 256,
        top_k: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.top_k = top_k
        self.num_classes = num_classes

        # --- Step 1: Query (Wo) and Key (Wp) projections ---
        self.w_o = nn.Linear(feature_dim, attn_dim, bias=False)
        self.w_p = nn.Linear(feature_dim, attn_dim, bias=False)

        # --- Step 2: Gating network g(F_0, F_n) = Sigmoid(Wg [F_0, F_n]) ---
        self.w_g = nn.Sequential(
            nn.Linear(feature_dim * 2, attn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 1),
            nn.Sigmoid(),
        )

        # --- Step 3: Self-Attention Block across X (5 tokens of 1024-D) ---
        self.sa_query = nn.Linear(feature_dim, attn_dim, bias=False)
        self.sa_key = nn.Linear(feature_dim, attn_dim, bias=False)
        self.sa_value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.w_x = nn.Linear(feature_dim, feature_dim, bias=False)

        # --- Step 4: Patch Classification Network ---
        # MLP -> LayerNorm -> ReLU -> Pooling -> MLP -> Softmax
        self.mlp1 = nn.Linear(feature_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        center_feats: torch.Tensor,
        neighbor_feats: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            center_feats: (B, 1024) features of central patches F_0,0
            neighbor_feats: (B, 8, 1024) features of 8 neighboring patches F_i,j
            neighbor_mask: Optional (B, 8) boolean mask (True = valid, False = missing/boundary)
            
        Returns:
            patch_probs: (B, 2) Softmax class probabilities [P(normal), P(tumor)]
            patch_logits: (B, 2) Raw unnormalized logits
            alpha: (B, 8) Softmax attention coefficients across 8 neighbors
            context_vector: (B, 1024) Consolidated context-enhanced patch representation
        """
        B, num_neighbors, D = neighbor_feats.shape
        assert num_neighbors == 8, f"Expected 8 neighbors, got {num_neighbors}"

        # --- Step 1: Compute Gated Attention Scores e_0,(i,j) ---
        # Query: (B, 1, attn_dim)
        query = self.w_o(center_feats).unsqueeze(1)
        # Key: (B, 8, attn_dim)
        key = self.w_p(neighbor_feats)

        # Dot product scaled by sqrt(d): (B, 8)
        sim_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1) / (self.attn_dim ** 0.5)

        # Relevance Gating g(F_0,0, F_i,j): (B, 8, 1)
        center_expanded = center_feats.unsqueeze(1).expand(-1, 8, -1)  # (B, 8, 1024)
        cat_feats = torch.cat([center_expanded, neighbor_feats], dim=-1)  # (B, 8, 2048)
        g_val = self.w_g(cat_feats).squeeze(-1)  # (B, 8)

        # Eq. (1): e_0,(i,j) = g(F_0,0, F_i,j) * (query^T * key) / sqrt(d)
        e_scores = g_val * sim_scores  # (B, 8)

        # Mask out boundary/missing neighbors with safe negative value for fp16
        if neighbor_mask is not None:
            mask_val = -1e4 if e_scores.dtype == torch.float16 else -1e9
            e_scores = e_scores.masked_fill(~neighbor_mask, mask_val)

        # Eq. (2): Softmax attention coefficients alpha_0,(i,j)
        alpha = F.softmax(e_scores, dim=-1)  # (B, 8)

        # --- Step 2: Select Top-K (K=4) Neighbors ---
        # Top-4 selection per central patch
        top_weights, top_indices = torch.topk(alpha, k=self.top_k, dim=-1)  # (B, 4)

        # Gather the selected neighbor feature vectors: (B, 4, 1024)
        batch_idx = torch.arange(B, device=center_feats.device).unsqueeze(1).expand(-1, self.top_k)
        selected_neighbors = neighbor_feats[batch_idx, top_indices]  # (B, 4, 1024)

        # Eq. (3): Scale selected neighbor features by their attention coefficients
        h_k = top_weights.unsqueeze(-1) * selected_neighbors  # (B, 4, 1024)

        # Eq. (4): X = Concat(F_0,0, H_1, H_2, H_3, H_4) -> (B, 5, 1024)
        x_central = center_feats.unsqueeze(1)  # (B, 1, 1024)
        X = torch.cat([x_central, h_k], dim=1)  # (B, 5, 1024)

        # --- Step 3: Self-Attention across X (Eq. 5) ---
        sa_q = self.sa_query(X)  # (B, 5, attn_dim)
        sa_k = self.sa_key(X)    # (B, 5, attn_dim)
        sa_v = self.sa_value(X)  # (B, 5, 1024)

        sa_scores = torch.bmm(sa_q, sa_k.transpose(1, 2)) / (self.attn_dim ** 0.5)  # (B, 5, 5)
        sa_weights = F.softmax(sa_scores, dim=-1)  # (B, 5, 5)
        beta_x = torch.bmm(sa_weights, sa_v)      # (B, 5, 1024)

        # Eq. (6): R = sigma(Wx * beta * X)
        R = F.relu(self.w_x(beta_x))  # (B, 5, 1024)

        # --- Step 4: Patch Classification Head ---
        # MLP -> LayerNorm -> ReLU
        flat_r = self.mlp1(R)  # (B, 5, 512)
        flat_r = self.norm1(flat_r)
        flat_r = F.relu(flat_r)

        # Consolidate 5 tokens via pooling across token dimension -> (B, 512)
        pooled = torch.mean(flat_r, dim=1)

        # Another MLP -> Logits & Softmax -> (B, 2)
        logits = self.mlp2(pooled)
        probs = F.softmax(logits, dim=-1)

        # Context-consolidated representation (used for slide-level aggregation)
        context_vector = torch.mean(R, dim=1)  # (B, 1024)

        return probs, logits, alpha, context_vector
