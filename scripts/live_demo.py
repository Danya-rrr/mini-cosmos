"""
Live CARLA prediction demo - real-time World Model inference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image
import cv2
import time
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API not found!")
    print("Add CARLA to PYTHONPATH:")
    print("  set PYTHONPATH=%PYTHONPATH%;C:/carla/PythonAPI/carla")
    sys.exit(1)

from src.models.vae import VAE, VAEConfig
from src.models.world_model import WorldModel, WorldModelConfig


class LivePredictor:
    
    def __init__(self, vae, world_model, context_length=8, image_size=(256, 256), device='cuda'):
        self.vae = vae.to(device)
        self.world_model = world_model.to(device)
        self.vae.eval()
        self.world_model.eval()
        
        self.context_length = context_length
        self.image_size = image_size
        self.device = device
        
        self.frame_buffer = deque(maxlen=context_length)
        self.action_buffer = deque(maxlen=context_length)
    
    def preprocess_frame(self, image):
        image = cv2.resize(image, self.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor * 2 - 1
        return tensor
    
    def add_frame(self, image, action=None):
        tensor = self.preprocess_frame(image)
        self.frame_buffer.append(tensor)
        
        if action is None:
            action = np.array([0.0, 0.0, 0.0])
        self.action_buffer.append(torch.from_numpy(action).float())
    
    def has_enough_context(self):
        return len(self.frame_buffer) >= self.context_length
    
    @torch.no_grad()
    def predict_next_frames(self, num_future=4, pred_step=1):
        if not self.has_enough_context():
            return []
        
        context = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        actions = torch.stack(list(self.action_buffer)).unsqueeze(0).to(self.device)
        
        B, T = context.shape[:2]
        context_flat = context.view(B * T, *context.shape[2:])
        latents = self.vae.get_latent(context_flat, deterministic=True)
        latents = latents.view(B, T, *latents.shape[1:])
        
        total_frames = num_future * pred_step
        predictions = []
        current_latents = latents.clone()
        
        for i in range(total_frames):
            input_latents = current_latents[:, -self.context_length:]
            
            output = self.world_model(input_latents, actions)
            pred_latent = output['pred_latent']
            
            if (i + 1) % pred_step == 0:
                pred_frame = self.vae.decode(pred_latent)
                pred_frame = (pred_frame + 1) / 2
                pred_frame = pred_frame.clamp(0, 1)
                
                pred_np = pred_frame[0].permute(1, 2, 0).cpu().numpy()
                pred_np = (pred_np * 255).astype(np.uint8)
                predictions.append(pred_np)
            
            current_latents = torch.cat([
                current_latents,
                pred_latent.unsqueeze(1)
            ], dim=1)
        
        return predictions
    
    def get_context_visualization(self):
        if len(self.frame_buffer) == 0:
            return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        last_frame = self.frame_buffer[-1]
        frame = (last_frame + 1) / 2
        frame = frame.clamp(0, 1)
        frame_np = frame.permute(1, 2, 0).numpy()
        frame_np = (frame_np * 255).astype(np.uint8)
        return frame_np


class CARLAInterface:
    
    def __init__(self, host='localhost', port=2000, image_size=(256, 256)):
        self.host = host
        self.port = port
        self.image_size = image_size
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
        self.latest_image = None
        self.latest_control = None
    
    def connect(self):
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        print(f"Connected! Map: {self.world.get_map().name}")
    
    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned vehicle at {spawn_point.location}")
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_size[0] * 2))
        camera_bp.set_attribute('image_size_y', str(self.image_size[1] * 2))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15)
        )
        
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_image)
        print("Camera attached!")
    
    def _on_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]
        
        if self.vehicle:
            control = self.vehicle.get_control()
            self.latest_control = np.array([control.throttle, control.steer, control.brake])
    
    def get_frame(self):
        return self.latest_image, self.latest_control
    
    def enable_autopilot(self, enable=True):
        if self.vehicle:
            self.vehicle.set_autopilot(enable)
            print(f"Autopilot: {'ON' if enable else 'OFF'}")
    
    def cleanup(self):
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("Cleaned up CARLA actors")


def create_visualization(context_frame, predictions, ground_truth=None, fps=0, pred_step=1):
    h, w = context_frame.shape[:2]
    num_pred = min(4, len(predictions)) if predictions else 0
    
    if ground_truth is not None:
        canvas_w = w * (2 + num_pred)
    else:
        canvas_w = w * (1 + num_pred)
    
    canvas = np.zeros((h + 40, canvas_w, 3), dtype=np.uint8)
    
    canvas[40:40+h, 0:w] = cv2.cvtColor(context_frame, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, "Context (t)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    for i, pred in enumerate(predictions[:num_pred]):
        x_offset = w * (i + 1)
        canvas[40:40+h, x_offset:x_offset+w] = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        frame_num = (i + 1) * pred_step
        seconds = frame_num / 20.0
        cv2.putText(canvas, f"t+{frame_num} (~{seconds:.1f}s)", (x_offset + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if ground_truth is not None:
        x_offset = w * (1 + num_pred)
        gt_resized = cv2.resize(ground_truth, (w, h))
        canvas[40:40+h, x_offset:x_offset+w] = gt_resized
        cv2.putText(canvas, "Ground Truth", (x_offset + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(canvas, f"FPS: {fps:.1f}", (canvas_w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas


def run_demo(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\nLoading models...")
    
    vae_path = Path(args.vae_checkpoint)
    wm_path = Path(args.world_model_checkpoint)
    
    if not vae_path.exists():
        print(f"[ERROR] VAE checkpoint not found: {vae_path}")
        return
    
    if not wm_path.exists():
        print(f"[ERROR] World Model checkpoint not found: {wm_path}")
        return
    
    vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=(64, 128, 256, 256)
    )
    vae = VAE(vae_config)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"  VAE loaded from epoch {vae_checkpoint['epoch']}")
    
    wm_checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    wm_config = WorldModelConfig(
        latent_dim=args.latent_dim,
        latent_size=32,
        context_length=args.context_length,
        action_dim=3,
        use_actions=True,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        latent_patch_size=4
    )
    world_model = WorldModel(wm_config)
    world_model.load_state_dict(wm_checkpoint['model_state_dict'])
    print(f"  World Model loaded from epoch {wm_checkpoint['epoch']}")
    
    predictor = LivePredictor(
        vae=vae,
        world_model=world_model,
        context_length=args.context_length,
        image_size=(256, 256),
        device=device
    )
    
    carla_interface = CARLAInterface(
        host=args.carla_host,
        port=args.carla_port,
        image_size=(256, 256)
    )
    
    video_writer = None
    
    try:
        carla_interface.connect()
        carla_interface.spawn_vehicle()
        carla_interface.enable_autopilot(True)
        
        print("\n" + "=" * 50)
        print("Live Demo Started!")
        print("Controls: Q=Quit, S=Screenshot, R=Record, SPACE=Pause")
        print("=" * 50 + "\n")
        
        print("Waiting for camera feed...")
        while carla_interface.latest_image is None:
            time.sleep(0.1)
        print("Camera feed received!")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        recording = False
        paused = False
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        cv2.namedWindow('Mini-Cosmos Live Demo', cv2.WINDOW_NORMAL)
        
        while True:
            loop_start = time.time()
            
            if not paused:
                image, control = carla_interface.get_frame()
                
                if image is not None:
                    ground_truth = image.copy()
                    predictor.add_frame(image, control)
                    
                    if predictor.has_enough_context():
                        predictions = predictor.predict_next_frames(
                            num_future=args.num_future,
                            pred_step=args.pred_step
                        )
                    else:
                        predictions = []
                    
                    context_vis = predictor.get_context_visualization()
                    
                    vis = create_visualization(
                        context_vis, predictions,
                        ground_truth if args.show_gt else None,
                        fps, pred_step=args.pred_step
                    )
                    
                    if recording and video_writer is not None:
                        video_writer.write(vis)
                    
                    cv2.imshow('Mini-Cosmos Live Demo', vis)
                    frame_count += 1
            
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = output_dir / f"screenshot_{timestamp}.png"
                cv2.imwrite(str(screenshot_path), vis)
                print(f"Screenshot saved: {screenshot_path}")
            elif key == ord('r'):
                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = output_dir / f"recording_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (vis.shape[1], vis.shape[0]))
                    recording = True
                    print(f"Recording started: {video_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print("Recording stopped")
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            
            elapsed = time.time() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        carla_interface.cleanup()
        print("Demo ended")


def main():
    parser = argparse.ArgumentParser(description='Live CARLA Demo')
    
    parser.add_argument('--carla_host', type=str, default='localhost')
    parser.add_argument('--carla_port', type=int, default=2000)
    
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--context_length', type=int, default=8)
    parser.add_argument('--num_future', type=int, default=4)
    parser.add_argument('--pred_step', type=int, default=1)
    
    parser.add_argument('--vae_checkpoint', type=str, default='./outputs/checkpoints/vae_best.pt')
    parser.add_argument('--world_model_checkpoint', type=str, default='./outputs/checkpoints/world_model_best.pt')
    
    parser.add_argument('--show_gt', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/live_demo')
    
    args = parser.parse_args()
    run_demo(args)


if __name__ == '__main__':
    main()