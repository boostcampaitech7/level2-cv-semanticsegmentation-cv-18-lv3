import os
from datetime import datetime

def create_project_structure(base_dir: str="./"):
    dirs = [
        f"{base_dir}/src",
        f"{base_dir}/src/utils",
        f"{base_dir}/src/models",
        f"{base_dir}/src/datasets",
        f"{base_dir}/data",
        f"{base_dir}/configs",
        f"{base_dir}/outputs",
        f"{base_dir}/wandb",
        f"{base_dir}/etc",
        f"{base_dir}/etc/streamlit",
        f"{base_dir}/etc/EDA",
        f"{base_dir}/etc/visualization"
    ]
    
    # Create directories
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Create main.py and configuration files in the base directory
    open(f"{base_dir}/main.py", 'a').close()
    for config_file in ["base.yaml", "path.yaml", "train.yaml"]:
        open(f"{base_dir}/configs/{config_file}", 'a').close()
    
    # Create source files in src
    for src_file in ["train.py", "inference.py"]:
        open(f"{base_dir}/src/{src_file}", 'a').close()
    
    # Create utility files in src/utils
    for util_file in ["metrics.py", "trainer.py", "wandb_logger.py"]:
        open(f"{base_dir}/src/utils/{util_file}", 'a').close()
    
    # Create model utility file in src/models
    open(f"{base_dir}/src/models/model_utils.py", 'a').close()
    
    # Create dataset files in src/datasets
    for dataset_file in ["dataset.py", "dataloader.py"]:
        open(f"{base_dir}/src/datasets/{dataset_file}", 'a').close()

    print("Directory structure created.")

# Run the function to create the directory structure
create_project_structure()