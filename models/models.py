# -*- coding: utf-8 -*-
"""models.ipynb

Original file is located at
    https://colab.research.google.com/drive/1CWYRv4zSNbURY9TzaDglC7qR6hz9q43P
"""

import timm
from timm.data import resolve_data_config
from .modifyModelBuilder import *
from .attentionLayers import *

# Pretrained-Model on ImageNet1K
timm_model = {
    'ResNet50': 'resnet50',
    'ResNeXt50': 'resnext50_32x4d.tv_in1k',
    'ConvNeXt-Tiny': 'convnext_tiny.fb_in1k',
    'ConvNeXtV2-Tiny': 'convnextv2_tiny.fcmae_ft_in1k',
    'InceptionNeXt-Tiny': 'inception_next_tiny.sail_in1k',
    'DeiT-Small': 'deit_small_patch16_224.fb_in1k',
    'SwinTransformer-Tiny': 'swin_tiny_patch4_window7_224.ms_in1k'
}

def build_baseline_model(model_name, num_classes, device, freeze_backbone=False):
    if model_name not in timm_model:
        raise ValueError(f"Model {model_name} not found. Available models: {list(timm_model.keys())}")

    timm_weight = timm_model[model_name]

    model = timm.create_model(
        timm_weight,
        pretrained=True,
        num_classes=num_classes
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model.to(device)

def build_modify_convnext(model_name, num_classes, device, target_stages=[0,1,2,3], freeze_backbone=False, modify_type='A1'):
    assert model_name == 'ConvNeXt-Tiny', "Only ConvNeXt-Tiny is supported"
    assert modify_type in ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'], "Invalid model type"
    timm_weight = timm_model[model_name]

    builder = ModifyConvNeXtBuilder(
        timm_weight=timm_weight,
        num_classes=num_classes,
        pretrained=True,
        device=device
    )

    reduction = 16
    min_hidden_channels = 48
    default_min_hidden_channels = 8

    cbam_pre_post_kwargs = {'reduction': reduction, 'min_hidden_channels': min_hidden_channels, 'kernel_size': 7, 'act_layer': nn.GELU}
    cbam_middle_kwargs = {'reduction': reduction, 'min_hidden_channels': default_min_hidden_channels, 'kernel_size': 7, 'act_layer': nn.GELU}

    se_pre_post_kwargs = {'reduction': reduction, 'min_hidden_channels': min_hidden_channels, 'act_layer': nn.GELU}
    se_middle_kwargs = {'reduction': reduction, 'min_hidden_channels': default_min_hidden_channels, 'act_layer': nn.GELU}

    ca_pre_post_kwargs = {'reduction': reduction, 'min_hidden_channels': min_hidden_channels, 'act_layer': nn.GELU}
    ca_middle_kwargs = {'reduction': reduction, 'min_hidden_channels': default_min_hidden_channels, 'act_layer': nn.GELU}

    simam_default_kwargs = {'e_lambda': 1e-04}

    if modify_type == 'A1':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=CBAM,
                    middle_attn_layer=None,
                    post_attn_layer=None,
                    pre_attn_kwargs=cbam_pre_post_kwargs,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'A2':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=CBAM,
                    post_attn_layer=None,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=cbam_middle_kwargs,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'A3':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=None,
                    post_attn_layer=CBAM,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=cbam_pre_post_kwargs
                )
                .get_model()
        )
    elif modify_type == 'B1':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=SE,
                    middle_attn_layer=None,
                    post_attn_layer=None,
                    pre_attn_kwargs=se_pre_post_kwargs,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'B2':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=SE,
                    post_attn_layer=None,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=se_middle_kwargs,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'B3':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=None,
                    post_attn_layer=SE,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=se_pre_post_kwargs
                )
                .get_model()
        )
    elif modify_type == 'C1':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=CoordinateAttention,
                    middle_attn_layer=None,
                    post_attn_layer=None,
                    pre_attn_kwargs=ca_pre_post_kwargs,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'C2':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=CoordinateAttention,
                    post_attn_layer=None,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=ca_middle_kwargs,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'C3':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=None,
                    post_attn_layer=CoordinateAttention,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=ca_pre_post_kwargs
                )
                .get_model()
        )
    elif modify_type == 'D1':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=SimAM,
                    middle_attn_layer=None,
                    post_attn_layer=None,
                    pre_attn_kwargs=simam_default_kwargs,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'D2':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=SimAM,
                    post_attn_layer=None,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=simam_default_kwargs,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'D3':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=None,
                    post_attn_layer=SimAM,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=None,
                    post_attn_kwargs=simam_default_kwargs
                )
                .get_model()
        )
    elif modify_type == 'D4':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=SimAM,
                    middle_attn_layer=SimAM,
                    post_attn_layer=None,
                    pre_attn_kwargs=simam_default_kwargs,
                    middle_attn_kwargs=simam_default_kwargs,
                    post_attn_kwargs=None
                )
                .get_model()
        )
    elif modify_type == 'D5':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=None,
                    middle_attn_layer=SimAM,
                    post_attn_layer=SimAM,
                    pre_attn_kwargs=None,
                    middle_attn_kwargs=simam_default_kwargs,
                    post_attn_kwargs=simam_default_kwargs
                )
                .get_model()
        )
    elif modify_type == 'D6':
        model = (builder
                .freeze_backbone(freeze_backbone)
                .modify_token_mixer_stage(
                    stage_indices=target_stages,
                    new_stage_class=ConvNeXtBlockModify,
                    only_last_block=True,
                    pre_attn_layer=SimAM,
                    middle_attn_layer=SimAM,
                    post_attn_layer=SimAM,
                    pre_attn_kwargs=simam_default_kwargs,
                    middle_attn_kwargs=simam_default_kwargs,
                    post_attn_kwargs=simam_default_kwargs
                )
                .get_model()
        )
    return model

def get_transform_config(model):
    data_config = resolve_data_config({}, model=model)

    transform_config = {
        'input_size': data_config['input_size'],
        'img_size': data_config['input_size'][1:],
        'crop_pct': data_config['crop_pct'],
        'interpolation': data_config['interpolation'],
        'mean': data_config['mean'],
        'std': data_config['std']
    }
    return transform_config