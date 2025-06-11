import torch
import cv2
import numpy as np
import os
from glob import glob

def run_inference(detector, input_path, model_path=None):
    if model_path:
        detector.load_model(model_path)
    
    detector.set_model_mode('eval')
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return process_video(detector, input_path)
        else:
            return process_single_image(detector, input_path)
    
    elif os.path.isdir(input_path):
        return process_image_directory(detector, input_path)
    
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def process_single_image(detector, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    result = detector.detect(image)
    
    return {
        'images': [{
            'path': image_path,
            'image': image,
            'detections': result['detections'],
            'enhanced_image': result['enhanced_image'],
            'candidate_regions': result['candidate_regions']
        }],
        'total_detections': len(result['detections'])
    }

def process_image_directory(detector, directory_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(directory_path, ext)))
        image_paths.extend(glob(os.path.join(directory_path, ext.upper())))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {directory_path}")
    
    results = []
    total_detections = 0
    
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Warning: Could not load {image_path}")
                continue
            
            result = detector.detect(image)
            
            results.append({
                'path': image_path,
                'image': image,
                'detections': result['detections'],
                'enhanced_image': result['enhanced_image'],
                'candidate_regions': result['candidate_regions']
            })
            
            total_detections += len(result['detections'])
            
            print(f"Processed {os.path.basename(image_path)}: {len(result['detections'])} detections")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    return {
        'images': results,
        'total_detections': total_detections,
        'processed_count': len(results)
    }

def process_video(detector, video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    results = []
    total_detections = 0
    frame_idx = 0
    
    print(f"Processing video: {frame_count} frames at {fps} FPS")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        result = detector.detect(gray_frame)
        
        results.append({
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'image': gray_frame,
            'detections': result['detections'],
            'enhanced_image': result['enhanced_image'],
            'candidate_regions': result['candidate_regions']
        })
        
        total_detections += len(result['detections'])
        
        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}/{frame_count}")
        
        frame_idx += 1
    
    cap.release()
    
    return {
        'video_info': {
            'path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height
        },
        'frames': results,
        'total_detections': total_detections
    }

def save_inference_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if 'images' in results:
        save_image_results(results['images'], output_dir)
    
    elif 'frames' in results:
        save_video_results(results, output_dir)
    
    summary_path = os.path.join(output_dir, 'inference_summary.txt')
    save_inference_summary(results, summary_path)

def save_image_results(image_results, output_dir):
    for i, result in enumerate(image_results):
        image = result['image']
        detections = result['detections']
        
        output_image = draw_detections_on_image(image, detections)
        
        filename = f"result_{i:04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_image)

def save_video_results(video_results, output_dir):
    frames = video_results['frames']
    video_info = video_results['video_info']
    
    output_video_path = os.path.join(output_dir, 'detection_results.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        video_info['fps'],
        (video_info['width'], video_info['height'])
    )
    
    for frame_result in frames:
        image = frame_result['image']
        detections = frame_result['detections']
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        output_frame = draw_detections_on_image(image, detections)
        out.write(output_frame)
    
    out.release()

def draw_detections_on_image(image, detections):
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = f'Vehicle: {confidence:.2f}'
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(output_image,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     (0, 0, 255), -1)
        
        cv2.putText(output_image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
    
    return output_image

def save_inference_summary(results, output_path):
    with open(output_path, 'w') as f:
        f.write("Night Vehicle Detection Inference Results\n")
        f.write("=" * 50 + "\n\n")
        
        if 'images' in results:
            f.write(f"Processed Images: {len(results['images'])}\n")
            f.write(f"Total Detections: {results['total_detections']}\n")
            f.write(f"Average Detections per Image: {results['total_detections'] / len(results['images']):.2f}\n\n")
            
            for i, result in enumerate(results['images']):
                f.write(f"Image {i+1}: {os.path.basename(result['path'])}\n")
                f.write(f"  Detections: {len(result['detections'])}\n")
                f.write(f"  Candidate Regions: {len(result['candidate_regions'])}\n\n")
        
        elif 'frames' in results:
            video_info = results['video_info']
            f.write(f"Video: {os.path.basename(video_info['path'])}\n")
            f.write(f"Frames: {video_info['frame_count']}\n")
            f.write(f"Duration: {video_info['frame_count'] / video_info['fps']:.2f} seconds\n")
            f.write(f"Total Detections: {results['total_detections']}\n")
            f.write(f"Average Detections per Frame: {results['total_detections'] / len(results['frames']):.2f}\n\n")