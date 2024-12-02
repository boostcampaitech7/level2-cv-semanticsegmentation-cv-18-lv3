# 🔗 HotCLIP
![KakaoTalk_Photo_2024-12-02-11-53-53](https://github.com/user-attachments/assets/8eaeb967-964b-4dca-a270-38d8a9320f19)

## 🤜 팀의 목표
- 협업 능력을 기르자!
- 지속 가능한 Baseline Code를 구축하자! (feat. 모델 이식)
- 이번 대회에 Vision Language Model을 활용해보자!

<br>

# 1. Project Overview (프로젝트 개요)
- Hand Bone Segmentation
뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

1. 질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

2. 수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

3. 의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

4. 의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

<br>
<br>

# 2. Team Members (팀원 및 팀 소개)

|                         곽기훈                         |                            김민지                            |                         김현기                         |                         이해강                         |                          장희진                          |                        홍유향                        |
|:------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------:|:--------------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
|  <img src="https://github.com/kkh090.png" width="250"> |   <img src="https://github.com/qzzloz.png" width="250">   |  <img src="https://github.com/hyeonrl98.png" width="250">   |  <img src="https://github.com/lazely.png" width="250">   | <img src="https://github.com/heeejini.png" width="250">| <img src="https://github.com/hyanghyanging.png" width="250"> |
| [kkh090](https://github.com/kkh090) | [qzzloz](https://github.com/qzzloz) | [hyeonrl98](https://github.com/hyeonrl98) | [lazely](https://github.com/lazely) | [heeejini](https://github.com/heeejini) | [hyanghyanging](https://github.com/hyanghyanging) |

<br>
<br>


# 3. Quick Start
```
pip install -r requirements.txt
```
<br>
<br>

### 프로젝트 구조
```plaintext
📦level2-cv-semanticsegmentation
 ┣ 📜main.py
 ┣ 📂configs
 ┃ ┣ 📜base.yaml
 ┃ ┣ 📜data.yaml
 ┃ ┣ 📜etc.yaml
 ┃ ┣ 📜path.yaml
 ┃ ┗ 📜train.yaml
 ┣ 📂data
 ┣ 📂etc
 ┃ ┣ 📜offline_augmentation.py
 ┃ ┣ 📂EDA
 ┃ ┃ ┗ 📜EDA.ipynb
 ┃ ┣ 📂dev
 ┃ ┃ ┗ 📜dev_utils.py
 ┃ ┣ 📂streamlit
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┗ 📜vis_eda.py
 ┃ ┗ 📂visualization
 ┃   ┣ 📂segmentation_visualizations
 ┃   ┣ 📜aug_visualization.ipynb
 ┃   ┣ 📜create_results.py
 ┃   ┗ 📜visualization.ipynb
 ┣ 📂outputs
 ┣ 📂src
 ┃ ┣ 📜train.py
 ┃ ┣ 📜ensemble.py
 ┃ ┣ 📜inference.py
 ┃ ┣ 📂datasets
 ┃ ┃ ┣ 📜dataloader.py
 ┃ ┃ ┗ 📜dataset.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜model_utils.py
 ┃ ┃ ┣ 📜models__.py
 ┃ ┃ ┗ 📜...
 ┃ ┗ 📂utils
 ┃   ┣ 📜alert.py
 ┃   ┣ 📜augmentation.py
 ┃   ┣ 📜logger.py
 ┃   ┣ 📜loss.py
 ┃   ┣ 📜metrics.py
 ┃   ┣ 📜rle_convert.py
 ┃   ┗ 📜trainer.py
 ┗ 📂wandb
```
<br>
<br>

# 4. Run
## Train
```
python main.py --mode train --configs_folder ./configs
```
## Inference
```
python main.py --mode inference --configs_folder ./outputs/{your_folder}
```
