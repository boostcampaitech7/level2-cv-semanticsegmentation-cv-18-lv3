import segmentation_models_pytorch as smp

smp_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
}

def get_smp_model(model_config, num_classes):

    # 모델 이름 분리: smp_unet_resnet34 -> ("unet", "resnet34")
    _, model_key, encoder_name = model_config["name"].split('_')
    if model_key not in smp_models:
        raise ValueError(f"Unknown SMP model: {model_key}")
    
    # SMP 모델 초기화
    model = smp_models[model_key](encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    return model