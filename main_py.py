import torch
import argparse
from src.utils.config_loader import load_config
from src.detection.detector import VehicleDetector
from src.utils.visualization import save_detection_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], required=True)
    parser.add_argument('--input', help='Input image/video path for inference')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--model', help='Model checkpoint path')
    args = parser.parse_args()

    config = load_config(args.config)
    detector = VehicleDetector(config)

    if args.mode == 'train':
        from experiments.train_model import train_model
        train_model(detector, config)
    
    elif args.mode == 'eval':
        from experiments.evaluate_model import evaluate_model
        evaluate_model(detector, config, args.model)
    
    elif args.mode == 'inference':
        from experiments.inference import run_inference
        results = run_inference(detector, args.input, args.model)
        save_detection_results(results, args.output)

if __name__ == "__main__":
    main()