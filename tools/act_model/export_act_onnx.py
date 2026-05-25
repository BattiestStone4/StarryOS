#!/usr/bin/env python3
"""Export a simplified ACT-like model to ONNX format."""

import torch
import torch.nn as nn
import numpy as np
import os


class SimpleACT(nn.Module):
    """Minimal ACT-style model: CNN backbone + transformer encoder-decoder + action head."""

    def __init__(self, hidden_dim=256, state_dim=14, action_dim=14,
                 num_encoder_layers=4, num_decoder_layers=6,
                 nhead=8, dim_feedforward=2048, num_queries=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Projections
        self.img_proj = nn.Linear(256, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, image, state):
        # image: (B, 3, 480, 640), state: (B, 14)
        B = image.shape[0]

        # CNN backbone
        feat = self.backbone(image)           # (B, 256, 1, 1)
        feat = feat.flatten(2).transpose(1, 2)  # (B, 1, 256)
        feat = self.img_proj(feat)             # (B, 1, hidden_dim)

        # State projection -> (B, 1, hidden_dim)
        state_tok = self.state_proj(state).unsqueeze(1)

        # Concatenate image token + state token as encoder input
        tokens = torch.cat([feat, state_tok], dim=1)  # (B, 2, hidden_dim)
        memory = self.encoder(tokens)                   # (B, 2, hidden_dim)

        # Decode with learnable queries
        queries = self.query_embed.expand(B, -1, -1)   # (B, 100, hidden_dim)
        decoded = self.decoder(queries, memory)         # (B, 100, hidden_dim)

        # Predict actions
        actions = self.action_head(decoded)             # (B, 100, 14)
        return actions


def main():
    # Reproducible weights
    torch.manual_seed(42)
    np.random.seed(42)

    model = SimpleACT()
    model.eval()

    # Dummy inputs
    dummy_image = torch.randn(1, 3, 480, 640)
    dummy_state = torch.randn(1, 14)

    # Export path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    onnx_path = os.path.join(project_root, 'rootfs_overlay', 'models', 'act.onnx')

    print(f"Exporting SimpleACT model to: {onnx_path}")

    torch.onnx.export(
        model,
        (dummy_image, dummy_state),
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image", "state"],
        output_names=["actions"],
        dynamic_axes=None,
    )
    print("ONNX export complete.")

    # Try to simplify with onnx-simplifier
    try:
        import onnx
        from onnxsim import simplify

        print("Simplifying with onnx-simplifier...")
        onnx_model = onnx.load(onnx_path)
        onnx_model_simplified, check = simplify(onnx_model)
        if check:
            onnx.save(onnx_model_simplified, onnx_path)
            print("Simplified model saved.")
        else:
            print("WARNING: Simplified model failed validation, keeping original.")
    except ImportError:
        print("onnx-simplifier not available, skipping simplification.")
    except Exception as e:
        print(f"onnx-simplifier failed: {e}, keeping original export.")

    # Verify
    import onnx
    model_check = onnx.load(onnx_path)
    onnx.checker.check_model(model_check)
    print("ONNX model verification passed.")

    # Print model size
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
