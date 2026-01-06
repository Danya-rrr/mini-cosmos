"""
Convert nuScenes dataset to training format.
"""

import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_nuscenes(input_dir='./data/external/nuscenes', output_dir='./data/processed/nuscenes',
                     target_size=(256, 256), camera='CAM_FRONT'):
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Preparing nuScenes dataset")
    print("=" * 50)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Camera: {camera}")
    print()
    
    version_dir = input_path / 'v1.0-mini'
    
    with open(version_dir / 'scene.json', 'r') as f:
        scenes = json.load(f)
    print(f"Found {len(scenes)} scenes")
    
    with open(version_dir / 'sample.json', 'r') as f:
        samples = json.load(f)
    
    with open(version_dir / 'sample_data.json', 'r') as f:
        sample_data_list = json.load(f)
    
    sample_by_token = {s['token']: s for s in samples}
    sample_data_by_token = {sd['token']: sd for sd in sample_data_list}
    
    sample_data_by_sample = {}
    for sd in sample_data_list:
        sample_token = sd.get('sample_token', '')
        if sample_token:
            if sample_token not in sample_data_by_sample:
                sample_data_by_sample[sample_token] = []
            sample_data_by_sample[sample_token].append(sd)
    
    total_frames = 0
    
    for scene_idx, scene in enumerate(tqdm(scenes, desc="Processing scenes")):
        scene_dir = output_path / f"scene_{scene_idx:04d}"
        scene_dir.mkdir(exist_ok=True)
        
        first_sample_token = scene['first_sample_token']
        
        scene_samples = []
        current_token = first_sample_token
        
        while current_token:
            sample = sample_by_token.get(current_token)
            if sample is None:
                break
            scene_samples.append(sample)
            current_token = sample.get('next', '')
            if not current_token:
                break
        
        frame_metadata = []
        frame_idx = 0
        
        for sample in scene_samples:
            sample_token = sample['token']
            sample_data_items = sample_data_by_sample.get(sample_token, [])
            
            cam_data = None
            for sd in sample_data_items:
                filename = sd.get('filename', '')
                if camera in filename:
                    cam_data = sd
                    break
            
            if not cam_data:
                continue
            
            img_filename = cam_data['filename']
            img_path = input_path / img_filename
            
            if not img_path.exists():
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size, Image.LANCZOS)
                
                output_img_path = scene_dir / f"frame_{frame_idx:05d}.png"
                img.save(output_img_path)
                
                frame_metadata.append({
                    'frame_id': frame_idx,
                    'timestamp': cam_data.get('timestamp', 0),
                    'original_file': img_filename,
                    'sample_token': sample_token
                })
                
                frame_idx += 1
                total_frames += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        scene_metadata = {
            'scene_id': scene_idx,
            'scene_name': scene['name'],
            'description': scene.get('description', ''),
            'num_frames': frame_idx,
            'camera': camera,
            'frames': frame_metadata
        }
        
        with open(scene_dir / 'metadata.json', 'w') as f:
            json.dump(scene_metadata, f, indent=2)
    
    print()
    print("=" * 50)
    print(f"Done! {len(scenes)} scenes, {total_frames} frames")
    print("=" * 50)


if __name__ == '__main__':
    prepare_nuscenes()