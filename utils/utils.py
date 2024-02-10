import torch
import os
import gc

def get_saved_embedding(embedding_filename, device='cuda'): # this is hardcoded and should possibly be changed
    full_embedding = torch.load(embedding_filename, map_location=torch.device(device))
    weights = full_embedding.weight.data

    embedding = torch.nn.Embedding.from_pretrained(weights)
    embedding.requires_grad_(False) # Freeze or unfreeze)
    return embedding

def make_dir_if_none(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

def clear_cuda_memory(obj):
    del obj
    obj = None
    gc.collect()
    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()