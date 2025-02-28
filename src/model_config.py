import torch
import argparse
from argparse import Namespace

def get_model_config(args:Namespace):
    """
    Returns the model configuration for the given model type and precision
    """
    model_type = args.type
    custom_num_layer=args.custom_num_layer
    custom_gbs=args.custom_gbs
    precision=args.precision
    
    if model_type == "gpt2XL": 
        model_config = {"hidden_size": torch.tensor([1600]).float(), # also known as d_model 4x
                    "sequence_length": torch.tensor([1024]).float(), # 2x
                    "num_layers": torch.tensor([custom_num_layer]).float(),   # 48
                    "vocab_size":torch.tensor([50257]).float(),
                    "num_attention_heads": torch.tensor([16]).float(), # 8x
                    "ffn_hidden_size": torch.tensor([1600*4]).float(),
                    "type": "gpt2XL",
                    "precision":torch.tensor(precision).float()}
        gbs = custom_gbs
    elif model_type == "llama2_13B": # LLAMA2-13B
        model_config = {"hidden_size": torch.tensor([5120]).float(),
                    "sequence_length": torch.tensor([4096]).float(), 
                    "num_layers": torch.tensor([40]).float(), 
                    "vocab_size":torch.tensor([32000]).float(),
                    "num_attention_heads": torch.tensor([40]).float(),
                    "ffn_hidden_size": torch.tensor([13824]).float(),
                    "type": "llama2_13B",
                    "precision":torch.tensor(precision).float()}
        gbs = custom_gbs 
    elif model_type == "llama2_13B_mini": # LLAMA2-13B
        model_config = {"hidden_size": torch.tensor([5120]).float(),
                    "sequence_length": torch.tensor([4096]).float(), 
                    "num_layers": torch.tensor([custom_num_layer]).float(), 
                    "vocab_size":torch.tensor([32000]).float(),
                    "num_attention_heads": torch.tensor([40]).float(),
                    "ffn_hidden_size": torch.tensor([13824]).float(),
                    "type": "llama2_13B",
                    "precision":torch.tensor(precision).float()}
        gbs = custom_gbs   
    else:
        assert False, "Model type not supported"

    return model_config, gbs