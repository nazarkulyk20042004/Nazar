import yaml
import os
import torch

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config = validate_config(config)
    config = set_device(config)
    
    return config

def validate_config(config):
    required_sections = ['model', 'training', 'data', 'preprocessing', 'detection']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    model_defaults = {
        'architecture': 'unet_attention',
        'input_size': [512, 512],
        'num_classes': 1,
        'encoder_channels': [64, 128, 256, 512],
        'decoder_channels': [256, 128, 64, 32],
        'attention_channels': 256,
        'activation': 'leaky_relu',
        'dropout_rate': 0.3
    }
    
    training_defaults = {
        'batch_size': 8,
        'epochs': 25,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'warmup_epochs': 5,
        'loss_weights': {
            'dice': 0.4,
            'focal': 0.3,
            'boundary': 0.2,
            'box': 0.1
        }
    }
    
    data_defaults = {
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'augmentation': True,
        'image_format': 'jpg'
    }
    
    preprocessing_defaults = {
        'clahe_clip_limit': 2.0,
        'gamma_correction': 2.2,
        'noise_reduction': True,
        'adaptive_threshold': True
    }
    
    detection_defaults = {
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'max_detections': 100,
        'cascade_scales': [64, 128, 256],
        'cascade_step': 4
    }
    
    config['model'] = {**model_defaults, **config.get('model', {})}
    config['training'] = {**training_defaults, **config.get('training', {})}
    config['data'] = {**data_defaults, **config.get('data', {})}
    config['preprocessing'] = {**preprocessing_defaults, **config.get('preprocessing', {})}
    config['detection'] = {**detection_defaults, **config.get('detection', {})}
    
    return config

def set_device(config):
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config['device'] = 'cpu'
    
    return config

def save_config(config, save_path):
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)

def update_config(config, updates):
    for key, value in updates.items():
        if '.' in key:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    
    return config

def create_experiment_config(base_config, experiment_name, modifications=None):
    config = base_config.copy()
    
    if modifications:
        config = update_config(config, modifications)
    
    config['experiment_name'] = experiment_name
    config['output_dir'] = f"results/{experiment_name}"
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    config_save_path = os.path.join(config['output_dir'], 'config.yaml')
    save_config(config, config_save_path)
    
    return config