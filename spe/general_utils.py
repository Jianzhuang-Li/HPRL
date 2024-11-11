import torch
import os
import time

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")
    
    def __getstate__(self):
        return self
    
    def __setstate__(self, d):
        self = d


class TensorModule(torch.nn.Module):
    """ A dummy module that wraps a single tensor and allows it to be handled like a network.

    Args:
        t (torch.Tensor): A single tensor.
    """
    def __init__(self, t) -> None:
        super().__init__()
        self.t = torch.nn.Parameter(t)

    def forward(self, *args, **kwargs):
        return self.t
    

class WeightSaver:
    def __init__(self, base_path=None):
        self.base_path = base_path

    def create_timestamp_folder(self, base_path=None):
        assert self.base_path is not None or base_path is not None
        if base_path is None:
            base_path = self.base_path
        timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        folder_path = os.path.join(base_path, timestamp)
        
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory: {e}")
        
        return folder_path

    def get_latest_folder(self):
        if self.base_path is None:
            print("Base path has not been set.")
            return
        
        folders = [f.path for f in os.scandir(self.base_path) if f.is_dir()]
        latest_folder = max(folders, key=lambda x: os.path.basename(x)) if folders else None
        
        return latest_folder
