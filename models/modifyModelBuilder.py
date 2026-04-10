# -*- coding: utf-8 -*-
"""modifyModelBuilder.ipynb

Original file is located at
    https://colab.research.google.com/drive/1gZOXv58n4Dl6g6Cqs5PgF4g0okQpAt_I
"""

import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_
from typing import List, Union, Optional, Dict, Any, Type
import inspect

class MlpWithMiddleAttention(nn.Module):
    def __init__(self, original_mlp: nn.Module, middle_attn: nn.Module, use_conv_mlp: bool = False):
        super().__init__()
        self.fc1 = original_mlp.fc1
        self.act = original_mlp.act
        self.drop1 = original_mlp.drop1
        self.norm = getattr(original_mlp, 'norm', nn.Identity())

        self.middle_attn = middle_attn

        self.fc2 = original_mlp.fc2
        self.drop2 = original_mlp.drop2
        self.use_conv_mlp = use_conv_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)

        if not isinstance(self.middle_attn, nn.Identity):
            if not self.use_conv_mlp:
                x = x.permute(0, 3, 1, 2)
                x = self.middle_attn(x)
                x = x.permute(0, 2, 3, 1)
            else:
                x = self.middle_attn(x)

        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ConvNeXtBlockModify(nn.Module):
    def __init__(
            self,
            original_block: nn.Module,
            pre_attn_layer: Optional[Type[nn.Module]] = None,
            middle_attn_layer: Optional[Type[nn.Module]] = None,
            post_attn_layer: Optional[Type[nn.Module]] = None,
            pre_attn_kwargs: Optional[Dict[str, Any]] = None,
            middle_attn_kwargs: Optional[Dict[str, Any]] = None,
            post_attn_kwargs: Optional[Dict[str, Any]] = None
            ) -> None:
        super().__init__()
        """
            This layer modify from original ConvNeXtBlock to include pre, middle, and post attention.
        """
        self.conv_dw = original_block.conv_dw
        self.norm = original_block.norm
        dim = self.conv_dw.out_channels

        pre_attn_kwargs = pre_attn_kwargs or {}
        middle_attn_kwargs = middle_attn_kwargs or {}
        post_attn_kwargs = post_attn_kwargs or {}

        # Pre-Attention
        self.pre_attn = self._build_attn(pre_attn_layer, dim, pre_attn_kwargs)

        # Get Hidden Dimension from MLP
        mlp_fc1 = original_block.mlp.fc1
        if hasattr(mlp_fc1, 'out_features'):
            hidden_dim = mlp_fc1.out_features
        elif hasattr(mlp_fc1, 'out_channels'):
            hidden_dim = mlp_fc1.out_channels
        else:
            hidden_dim = dim * 4 # Default

        # Middle-Attention
        middle_attn = self._build_attn(middle_attn_layer, hidden_dim, middle_attn_kwargs)
        self.use_conv_mlp = getattr(original_block, 'use_conv_mlp', False)
        self.mlp = MlpWithMiddleAttention(original_block.mlp, middle_attn, self.use_conv_mlp)

        # Post-Attention
        self.post_attn = self._build_attn(post_attn_layer, dim, post_attn_kwargs)

        self.gamma = getattr(original_block, 'gamma', None)
        self.shortcut = original_block.shortcut
        self.drop_path = original_block.drop_path

    def _build_attn(self, layer_cls: Optional[Type[nn.Module]], dim: int, kwargs: Dict[str, Any]) -> nn.Module:
        if layer_cls is None:
            return nn.Identity()

        sig = inspect.signature(layer_cls.__init__)
        params = list(sig.parameters.keys())
        if 'self' in params: params.remove('self')
        if not params:
            return layer_cls(**kwargs)

        channel_names = ['in_channels', 'out_channels']
        dim_params_needed = [p for p in params if p in channel_names]

        accepts_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())

        if dim_params_needed:
            positional_dims = [dim for _ in range(len(dim_params_needed))]
            return layer_cls(*positional_dims, **kwargs)

        elif accepts_var_args:
            return layer_cls(dim, **kwargs)

        else:
            return layer_cls(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.conv_dw(x)

        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.pre_attn(x)
            x = self.mlp(x)
            x = self.post_attn(x)
        else:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)

            if not isinstance(self.pre_attn, nn.Identity):
                x = x.permute(0, 3, 1, 2)
                x = self.pre_attn(x)
                x = x.permute(0, 2, 3, 1)

            x = self.mlp(x)

            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            x = self.post_attn(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(input_tensor)
        return x

class ModifyConvNeXtBuilder:
    def __init__(self, timm_weight: str, num_classes: int = 9, pretrained: bool = True, device: str = 'cpu') -> None:
        self.device = device
        self.num_classes = num_classes

        self.model = timm.create_model(
            timm_weight,
            pretrained=pretrained,
            num_classes=num_classes
        ).to(self.device)

    def freeze_backbone(self, freeze: bool = True) -> "ModifyConvNeXtBuilder":
        for param in self.model.parameters():
            param.requires_grad = not freeze
        return self

    def modify_token_mixer_stage(self, stage_indices: Union[int, List[int]], new_stage_class: Type[nn.Module], only_last_block: bool = True, **kwargs: Any) -> "ModifyConvNeXtBuilder":
        for stage in self._get_stages(stage_indices):
            if len(stage.blocks) == 0:
                continue

            if only_last_block:
                indices_to_modify = [len(stage.blocks) - 1]
            else:
                indices_to_modify = range(len(stage.blocks))

            for i in indices_to_modify:
                original_block = stage.blocks[i]
                new_block = new_stage_class(original_block, **kwargs).to(self.device)

                if hasattr(new_block, 'pre_attn') and not isinstance(new_block.pre_attn, nn.Identity):
                    for name, m in new_block.pre_attn.named_modules():
                        self._init_weights(m, name=name)

                if hasattr(new_block, 'middle_attn') and not isinstance(new_block.middle_attn, nn.Identity):
                    for name, m in new_block.middle_attn.named_modules():
                        self._init_weights(m, name=name)

                if hasattr(new_block, 'post_attn') and not isinstance(new_block.post_attn, nn.Identity):
                    for name, m in new_block.post_attn.named_modules():
                        self._init_weights(m, name=name)

                stage.blocks[i] = new_block
        return self

    def _get_stages(self, stage_indices: Union[int, List[int]]) -> List[nn.Module]:
        if isinstance(stage_indices, int):
            stage_indices = [stage_indices]
        return [self.model.stages[i] for i in stage_indices]

    @staticmethod
    def _init_weights(module: nn.Module, name: Optional[str] = None, head_init_scale: float = 1.0) -> None:
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
                if name and 'head.' in name:
                    module.bias.data.mul_(head_init_scale)

            if name and 'head.' in name:
                module.weight.data.mul_(head_init_scale)

    def get_model(self) -> nn.Module:
        return self.model