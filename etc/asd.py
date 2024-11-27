import sys
sys.path.append("/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3")

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from torchscale.model.BEiT3 import BEiT3

from src.models.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
import src.models.beit3.utils as bu
from PIL import Image
import torch
import requests

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

# print(type(model.model.pixel_level_module.encoder))

# args = _get_base_config(img_size=224, drop_path_rate=0.15, vocab_size=64010)
# model.model.pixel_level_module.encoder = BEiT3(args)
# bu.custom_load_model_and_may_interpolate("/data/ephemeral/home/kwak/level2-cv-semanticsegmentation-cv-18-lv3/src/models/beit3/checkpoints/beit3_base_patch16_224.pth",
#                                           model.model.pixel_level_module.encoder,
#                                           "model|module",
#                                           "")

# print(type(model.model.pixel_level_module.encoder))

# raise
# 테스트할 이미지 로드
url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 이미지 전처리 (모델 입력 형식으로 변환)
inputs = image_processor(image, return_tensors="pt")

# x = model.model.pixel_level_module.encoder(textual_tokens=None, visual_tokens=inputs['pixel_values'])
# print(x['encoder_out'].shape)
## 여기까진 동작함

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
print(outputs['masks_queries_logits'].shape)
# 3. 후처리 (Semantic Segmentation 맵 생성)
predicted_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]
predicted_map = predicted_map.cpu().numpy()
print(predicted_map.shape)
# 4. 결과 출력
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(predicted_map, cmap="jet")
plt.title("Predicted Semantic Segmentation")
plt.show()