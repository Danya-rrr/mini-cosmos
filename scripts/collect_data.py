#!/usr/bin/env python3
"""
Mini-Cosmos: CARLA Data Collection Script
=========================================
Collects driving data from CARLA simulator with:
- Ego vehicle with autopilot
- NPC vehicles and pedestrians
- RGB camera recording
- Metadata (position, velocity, actions)

Usage:
    python scripts/collect_data.py --episodes 10 --frames 500
"""

import carla
import random
import time
import os
import json
import argparse
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class EpisodeConfig:
    """Episode configuration"""
    num_vehicles: int = 30
    num_pedestrians: int = 20
    frames_per_episode: int = 500
    image_width: int = 256
    image_height: int = 256
    fov: int = 90
    fps: int = 10
    weather_change: bool = True


@dataclass
class FrameData:
    """Single frame metadata"""
    frame_id: int
    timestamp: float
    ego_location: Tuple[float, float, float]
    ego_rotation: Tuple[float, float, float]
    ego_velocity: Tuple[float, float, float]
    control_throttle: float
    control_steer: float
    control_brake: float
    num_visible_vehicles: int
    num_visible_pedestrians: int
    weather: str


class CARLADataCollector:
    """
    Main class for collecting data from CARLA.
    
    Creates diverse urban scenes with:
    - Ego vehicle with autopilot
    - NPC vehicles (also with autopilot)
    - Pedestrians with AI controller
    - Various weather conditions
    """
    
    WEATHER_PRESETS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
    ]
    
    WEATHER_NAMES = [
        "ClearNoon", "CloudyNoon", "WetNoon", 
        "SoftRainNoon", "ClearSunset", "CloudySunset"
    ]
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        output_dir: str = './data/raw',
        config: Optional[EpisodeConfig] = None
    ):
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.config = config or EpisodeConfig()
        
        # CARLA objects
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.traffic_manager = None
        
        # Actors
        self.ego_vehicle = None
        self.camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        
        # Data
        self.image_queue = queue.Queue()
        self.current_weather_idx = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to CARLA server"""
        try:
            print(f"Connecting to CARLA at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            
            # Setup Traffic Manager
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.set_synchronous_mode(True)
            
            # Synchronous mode for stable FPS
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.config.fps
            self.world.apply_settings(settings)
            
            print("[OK] Connected to CARLA")
            return True
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def spawn_ego_vehicle(self) -> bool:
        """Spawn main vehicle with camera"""
        try:
            # Choose vehicle (Tesla Model 3)
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            
            # Random spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Enable autopilot
            self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
            # Autopilot settings
            self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 0)
            self.traffic_manager.distance_to_leading_vehicle(self.ego_vehicle, 2.0)
            self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, -10)
            
            print(f"[OK] Ego vehicle spawned: {vehicle_bp.id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to spawn ego vehicle: {e}")
            return False
    
    def attach_camera(self) -> bool:
        """Attach RGB camera to ego vehicle"""
        try:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            
            # Camera settings
            camera_bp.set_attribute('image_size_x', str(self.config.image_width))
            camera_bp.set_attribute('image_size_y', str(self.config.image_height))
            camera_bp.set_attribute('fov', str(self.config.fov))
            
            # Camera position (on hood, forward view)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-5)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.ego_vehicle
            )
            
            # Callback for saving images
            self.camera.listen(lambda image: self.image_queue.put(image))
            
            print(f"[OK] Camera attached ({self.config.image_width}x{self.config.image_height})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to attach camera: {e}")
            return False
    
    def spawn_npc_vehicles(self) -> int:
        """Spawn NPC vehicles with autopilot"""
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        # Filter only cars (4 wheels)
        vehicle_bps = [
            bp for bp in self.blueprint_library.filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4
        ]
        
        spawned = 0
        for spawn_point in spawn_points[:self.config.num_vehicles]:
            try:
                bp = random.choice(vehicle_bps)
                
                # Random color
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)
                
                vehicle = self.world.spawn_actor(bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                
                # Behavior variety
                self.traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, random.uniform(-20, 20)
                )
                
                self.npc_vehicles.append(vehicle)
                spawned += 1
                
            except Exception:
                continue
        
        print(f"[OK] NPC vehicles: {spawned}/{self.config.num_vehicles}")
        return spawned
    
    def spawn_pedestrians(self) -> int:
        """Spawn pedestrians with AI controller"""
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        controller_bp = self.blueprint_library.find('controller.ai.walker')
        
        # Get random spawn locations on sidewalks
        spawn_locations = []
        for _ in range(self.config.num_pedestrians * 2):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_locations.append(carla.Transform(loc))
        
        spawned = 0
        for spawn_loc in spawn_locations[:self.config.num_pedestrians]:
            try:
                walker_bp = random.choice(walker_bps)
                
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                walker = self.world.spawn_actor(walker_bp, spawn_loc)
                self.pedestrians.append(walker)
                
                # AI controller
                controller = self.world.spawn_actor(
                    controller_bp,
                    carla.Transform(),
                    attach_to=walker
                )
                self.pedestrian_controllers.append(controller)
                
                spawned += 1
                
            except Exception:
                continue
        
        # Start controllers
        self.world.tick()
        
        for controller in self.pedestrian_controllers:
            try:
                controller.start()
                dest = self.world.get_random_location_from_navigation()
                if dest:
                    controller.go_to_location(dest)
                controller.set_max_speed(random.uniform(1.0, 2.5))
            except Exception:
                continue
        
        print(f"[OK] Pedestrians: {spawned}/{self.config.num_pedestrians}")
        return spawned
    
    def set_weather(self, weather_idx: Optional[int] = None):
        """Set weather"""
        if weather_idx is None:
            weather_idx = random.randint(0, len(self.WEATHER_PRESETS) - 1)
        
        self.current_weather_idx = weather_idx
        self.world.set_weather(self.WEATHER_PRESETS[weather_idx])
        print(f"[OK] Weather: {self.WEATHER_NAMES[weather_idx]}")
    
    def get_frame_data(self, frame_id: int) -> FrameData:
        """Collect current frame metadata"""
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        control = self.ego_vehicle.get_control()
        
        # Count visible objects (within 50m)
        ego_loc = transform.location
        visible_vehicles = sum(
            1 for v in self.npc_vehicles
            if v.get_location().distance(ego_loc) < 50
        )
        visible_pedestrians = sum(
            1 for p in self.pedestrians
            if p.get_location().distance(ego_loc) < 50
        )
        
        return FrameData(
            frame_id=frame_id,
            timestamp=time.time(),
            ego_location=(transform.location.x, transform.location.y, transform.location.z),
            ego_rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            ego_velocity=(velocity.x, velocity.y, velocity.z),
            control_throttle=control.throttle,
            control_steer=control.steer,
            control_brake=control.brake,
            num_visible_vehicles=visible_vehicles,
            num_visible_pedestrians=visible_pedestrians,
            weather=self.WEATHER_NAMES[self.current_weather_idx]
        )
    
    def collect_episode(self, episode_id: int) -> bool:
        """Collect single episode"""
        episode_dir = self.output_dir / f"episode_{episode_id:04d}"
        images_dir = episode_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'episode_id': episode_id,
            'config': asdict(self.config),
            'start_time': datetime.now().isoformat(),
            'frames': []
        }
        
        print(f"\n{'='*50}")
        print(f"Episode {episode_id}: Collecting {self.config.frames_per_episode} frames...")
        print(f"{'='*50}")
        
        # Change weather
        if self.config.weather_change:
            self.set_weather()
        
        # Warmup
        for _ in range(30):
            self.world.tick()
        
        # Clear image queue
        while not self.image_queue.empty():
            self.image_queue.get()
        
        collected_frames = 0
        start_time = time.time()
        
        for frame_id in range(self.config.frames_per_episode):
            try:
                # Simulation step
                self.world.tick()
                
                # Get image
                try:
                    image = self.image_queue.get(timeout=2.0)
                except queue.Empty:
                    print(f"  Skip frame {frame_id}: timeout")
                    continue
                
                # Save image
                image_path = images_dir / f"frame_{frame_id:05d}.png"
                image.save_to_disk(str(image_path))
                
                # Collect metadata
                frame_data = self.get_frame_data(frame_id)
                metadata['frames'].append(asdict(frame_data))
                
                collected_frames += 1
                
                # Progress
                if (frame_id + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = collected_frames / elapsed
                    print(f"  Frame {frame_id + 1}/{self.config.frames_per_episode} ({fps_actual:.1f} FPS)")
                
            except Exception as e:
                print(f"  Error at frame {frame_id}: {e}")
                continue
        
        # Save metadata
        metadata['end_time'] = datetime.now().isoformat()
        metadata['total_frames'] = collected_frames
        metadata['duration_seconds'] = time.time() - start_time
        
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[OK] Episode {episode_id} done: {collected_frames} frames")
        return True
    
    def cleanup_actors(self):
        """Remove all spawned actors"""
        print("Cleaning up actors...")
        
        # Stop pedestrian controllers
        for controller in self.pedestrian_controllers:
            try:
                controller.stop()
            except Exception:
                pass
        
        # Destroy actors
        actors_to_destroy = (
            [self.camera, self.ego_vehicle] +
            self.npc_vehicles +
            self.pedestrian_controllers +
            self.pedestrians
        )
        
        for actor in actors_to_destroy:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        
        # Clear lists
        self.camera = None
        self.ego_vehicle = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        
        print("[OK] Actors cleaned up")
    
    def reset_world(self):
        """Reset world for new episode"""
        self.cleanup_actors()
        time.sleep(1.0)
        
        # Respawn actors
        self.spawn_ego_vehicle()
        self.attach_camera()
        self.spawn_npc_vehicles()
        self.spawn_pedestrians()
    
    def collect_dataset(self, num_episodes: int):
        """Collect full dataset"""
        print(f"\n{'#'*60}")
        print(f"Mini-Cosmos: CARLA Data Collection")
        print(f"{'#'*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Frames per episode: {self.config.frames_per_episode}")
        print(f"Total frames: {num_episodes * self.config.frames_per_episode}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'#'*60}\n")
        
        if not self.connect():
            return
        
        # Initial setup
        self.spawn_ego_vehicle()
        self.attach_camera()
        self.spawn_npc_vehicles()
        self.spawn_pedestrians()
        
        try:
            for episode_id in range(num_episodes):
                self.collect_episode(episode_id)
                
                # Reset for next episode
                if episode_id < num_episodes - 1:
                    self.reset_world()
                    
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user")
        finally:
            self.cleanup_actors()
            
            # Restore async mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        print("\n[OK] Data collection finished!")
        self._print_dataset_stats()
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        total_frames = 0
        total_size = 0
        
        for episode_dir in self.output_dir.iterdir():
            if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
                images_dir = episode_dir / 'images'
                if images_dir.exists():
                    frames = len(list(images_dir.glob('*.png')))
                    size = sum(f.stat().st_size for f in images_dir.glob('*.png'))
                    total_frames += frames
                    total_size += size
        
        print(f"\n{'='*40}")
        print("Dataset statistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Size: {total_size / (1024**3):.2f} GB")
        print(f"  Path: {self.output_dir}")
        print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description='Mini-Cosmos: CARLA Data Collection')
    
    # Main parameters
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes (default: 10)')
    parser.add_argument('--frames', type=int, default=500,
                        help='Frames per episode (default: 500)')
    parser.add_argument('--output', type=str, default='./data/raw',
                        help='Output directory')
    
    # Scene parameters
    parser.add_argument('--vehicles', type=int, default=30,
                        help='Number of NPC vehicles (default: 30)')
    parser.add_argument('--pedestrians', type=int, default=20,
                        help='Number of pedestrians (default: 20)')
    
    # Image parameters
    parser.add_argument('--width', type=int, default=256,
                        help='Image width (default: 256)')
    parser.add_argument('--height', type=int, default=256,
                        help='Image height (default: 256)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10)')
    
    # CARLA connection
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA port (default: 2000)')
    
    args = parser.parse_args()
    
    # Configuration
    config = EpisodeConfig(
        num_vehicles=args.vehicles,
        num_pedestrians=args.pedestrians,
        frames_per_episode=args.frames,
        image_width=args.width,
        image_height=args.height,
        fps=args.fps
    )
    
    # Start collection
    collector = CARLADataCollector(
        host=args.host,
        port=args.port,
        output_dir=args.output,
        config=config
    )
    
    collector.collect_dataset(args.episodes)


if __name__ == '__main__':
    main()