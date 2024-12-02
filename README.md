# 1. Project Overview (프로젝트 개요)
- Hand Bone Segmentation


# 2. Team Members (팀원 및 팀 소개)

|                         곽기훈                         |                            김민지                            |                         김현기                         |                         이해강                         |                          장희진                          |                        홍유향                        |
|:------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------:|:--------------------------------------------------------:|:----------------------------------------------------:|:----------------------------------------------------:|
|  <img src="https://github.com/kkh090.png" width="250"> |   <img src="https://github.com/qzzloz.png" width="250">   |  <img src="https://github.com/hyeonrl98.png" width="250">   |  <img src="https://github.com/lazely.png" width="250">   | <img src="https://github.com/heeejini.png" width="250">| <img src="https://github.com/hyanghyanging.png" width="250"> |
| [kkh090](https://github.com/kkh090) | [qzzloz](https://github.com/qzzloz) | [hyeonrl98](https://github.com/hyeonrl98) | [lazely](https://github.com/lazely) | [heeejini](https://github.com/heeejini) | [hyanghyanging](https://github.com/hyanghyanging) |

<br>
<br>


# 3. Quick Start
```
```

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

# 4. Run
## Train

## Inference
