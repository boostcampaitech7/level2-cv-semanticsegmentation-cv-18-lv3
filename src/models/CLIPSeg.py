import inspect
import torch
from typing import Any

from src.models.clipseg.general_utils import (
    TrainingLogger,
    get_attribute,
    filter_args,
    log,
    training_config_from_cli_args,
)


def get_clipseg() -> torch.nn.Module:
    clipseg_config = training_config_from_cli_args()
    model_cls = get_attribute(clipseg_config.model)
    _, model_args, _ = filter_args(clipseg_config, inspect.signature(model_cls).parameters)
    model = model_cls(**model_args)
    return model


def prepare_conditional(
        model: torch.nn.Module, data_x: Any, use_prompts: bool = False
    ) -> torch.Tensor:
    """
    Prepare conditional vector based on model type (CLIP or visual prompt).
    """
    if use_prompts:
        prompts = model.sample_prompts(data_x)
        cond = model.compute_conditional(prompts)
    else:
        cond = data_x
        if isinstance(cond, torch.Tensor):
            cond = cond.cuda()
    return cond
