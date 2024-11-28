
### 프로젝트 구조
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
