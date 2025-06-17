import torch
import logging
from pathlib import Path

def analyze_checkpoint(checkpoint_path, log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        filemode='w'
    )

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    logging.info("\nCheckpoint content keys:")
    for key in checkpoint.keys():
        logging.info(f"- {key}")
    
    logging.info("\nAnalyzing model state dict:")
    state_dict = checkpoint["model"]
    total_params = 0
    
    logging.info("\nParameter sizes:")
    sizes = {}
    for key, tensor in state_dict.items():
        param_size = tensor.numel() * tensor.element_size()
        sizes[key] = param_size
        total_params += param_size
        
    sorted_sizes = dict(sorted(sizes.items(), key=lambda x: x[1], reverse=True))
    
    for key, size in sorted_sizes.items():
        size_mb = size / (1024 * 1024)
        logging.info(f"{key}: {size_mb:.2f} MB")
        
    logging.info(f"\nTotal model state dict size: {total_params / (1024 * 1024):.2f} MB")
    
    if "optimizer" in checkpoint:
        try:
            if "state" in checkpoint["optimizer"]:
                opt_size = sum(sum(p.numel() * p.element_size() for p in state.values() if isinstance(p, torch.Tensor))
                            for state in checkpoint["optimizer"]["state"].values())
                logging.info(f"Optimizer state size: {opt_size / (1024 * 1024):.2f} MB")
            else:
                logging.info("Optimizer state dict has unexpected structure")
        except Exception as e:
            logging.info(f"Could not calculate optimizer size: {str(e)}")

# Usage:
checkpoint_path = "/path/to/checkpoint_best.pth"  # Replace with actual path
log_path = "/path/to/checkpoint_analysis.log"  # Replace with desired log path
analyze_checkpoint(checkpoint_path, log_path)