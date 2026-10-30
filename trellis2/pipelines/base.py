from typing import *
import torch
import torch.nn as nn
from .. import models


def _release_host_memory():
    """
    Hand freed host memory back to the system. glibc keeps what the CPU copies of the weights used after the models
    move to the GPU (about the size of the models), and the process would hold it until exit.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass    # not glibc


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", model_names: Optional[List[str]] = None, device: Optional[str] = None) -> "Pipeline":
        """
        Load a pretrained model.

        Args:
            model_names (list): The models to load, in place of the class's model_names_to_load.
            device (str): Move each model to this device as soon as it is loaded, so the host RAM holds one at a time.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        if model_names is None:
            model_names = getattr(cls, 'model_names_to_load', None)
        _models = {}
        for k, v in args['models'].items():
            if model_names is not None and k not in model_names:
                continue
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)
            if device is not None:
                _models[k].to(device)
                _release_host_memory()

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))