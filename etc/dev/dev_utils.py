from typing import Any
import shutil
import random
import os

def dev_paths_setting(path_config: dict[str, Any]):
    data_root = './data'
    new_data_name = 'small_dev_dataset'
    new_data_path = os.path.join(data_root, new_data_name)

    if new_data_name not in os.listdir(data_root):
        create_small_dataset(path_config, new_data_path)
    
    path_config['train_image_root'] = os.path.join(new_data_path, 'train/DCM')
    path_config['train_label_root'] = os.path.join(new_data_path, 'train/outputs_json')
    path_config['inference_image_root'] = os.path.join(new_data_path, 'test/DCM')
    path_config['inference_label_root'] = os.path.join(new_data_path, 'test/outputs_json')
    
    # return path_config
    
def create_small_dataset(config: dict[str, Any],
                         target_root: str='./data/small_dev_dataset', 
                         n_train_samples: int=10, 
                         m_test_samples: int=5,
                         is_random: bool=False) -> tuple[str, str, str, str]:
    
    train_img_dir = config['train_image_root']
    train_label_dir = config['train_label_root']
    test_img_dir = config['inference_image_root']
    
    target_train_img_dir = os.path.join(target_root, 'train/DCM')
    target_train_label_dir = os.path.join(target_root, 'train/outputs_json')
    
    target_test_img_dir = os.path.join(target_root, 'test/DCM')
    target_test_label_dir = os.path.join(target_root, 'test/outputs_json')
    
    os.makedirs(target_train_img_dir, exist_ok=True)
    os.makedirs(target_train_label_dir, exist_ok=True)
    os.makedirs(target_test_img_dir, exist_ok=True)
    os.makedirs(target_test_label_dir, exist_ok=True)
    
    if is_random:
        train_ids = random.sample(os.listdir(target_train_img_dir), n_train_samples)
        test_ids = random.sample(os.listdir(target_test_img_dir), m_test_samples)
    else:
        train_ids = sorted(os.listdir(train_img_dir))[:n_train_samples]
        test_ids = sorted(os.listdir(test_img_dir))[:m_test_samples]
    
    for train_id in train_ids:
        shutil.copytree(os.path.join(train_img_dir, train_id), os.path.join(target_train_img_dir, train_id))
        shutil.copytree(os.path.join(train_label_dir, train_id), os.path.join(target_train_label_dir, train_id))
        
    for test_id in test_ids:
        shutil.copytree(os.path.join(test_img_dir, test_id), os.path.join(target_test_img_dir, test_id))

    