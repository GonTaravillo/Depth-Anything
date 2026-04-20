import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))

# We will patch the classes dynamically to test sizes without modifying files
from depth_anything.dpt import DPTHead
from torchhub.facebookresearch_dinov2_main.vision_transformer import DinoVisionTransformer, Block, MemEffAttention
from functools import partial

def test_model(embed_dim, depth, num_heads, decoder_features, out_channels):
    encoder = DinoVisionTransformer(
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=0,
    )
    dim = encoder.blocks[0].attn.qkv.in_features
    decoder = DPTHead(1, dim, decoder_features, False, out_channels=out_channels, use_clstoken=False, act_layer=torch.nn.GELU)
    
    total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"Config: embed={embed_dim}, depth={depth}, heads={num_heads}, dec_feat={decoder_features}, out_chan={out_channels} => Params: {total:,}")

test_model(32, 4, 1, 8, [8,16,32,64])
test_model(64, 4, 1, 16, [16,32,64,64])
test_model(96, 4, 2, 24, [24,48,96,96])
test_model(96, 6, 2, 24, [24,48,96,96])
test_model(128, 4, 2, 32, [24,48,96,96])
test_model(128, 6, 2, 32, [24,48,96,128])
