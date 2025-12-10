import time
import queue
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import carla


class CarlaEnv:
    """
    Proste środowisko oparte o CARLA:
    - ego–pojazd z kamerą RGB,
    - opcjonalny autopilot ego–auta,
    - dodatkowe pojazdy NPC (traffic),
    - piesi (walkers) z AI-kontrolerem,
    - tryb synchroniczny (deterministyczny),
    - spectator w oknie gry ustawiony "jak w GTA" (kamera za autem).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: Optional[str] = None,
        fps: int = 20,
        image_width: int = 160,
        image_height: int = 90,
        fov: float = 90.0,
        autopilot: bool = True,
        num_traffic_vehicles: int = 5,
        num_pedestrians: int = 15,
    ):
        """
        :param host: adres hosta CARLI (default: localhost)
        :param port: port CARLI (default: 2000)
        :param town: nazwa mapy (np. "Town05"); jeśli None, używa aktualnej
        :param fps: docelowa liczba klatek na sekundę
        :param image_width: szerokość obrazu z kamery
        :param image_height: wysokość obrazu z kamery
        :param fov: field of view kamery
        :param autopilot: czy ego–pojazd ma jechać na autopilocie
        :param num_traffic_vehicles: ile pojazdów NPC zespawnować
        :param num_pedestrians: ilu pieszych zespawnować
        """
        self.host = host
        self.port = port
        self.town = town
        self.fps = fps
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.autopilot = autopilot
        self.num_traffic_vehicles = num_traffic_vehicles
        self.num_pedestrians = num_pedestrians

        # klient i świat
        self.client: carla.Client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)

        if self.town is not None:
            self.world: carla.World = self.client.load_world(self.town)
        else:
            self.world: carla.World = self.client.get_world()

        # zapamiętujemy oryginalne ustawienia świata, żeby je przywrócić na końcu
        self.original_settings: carla.WorldSettings = self.world.get_settings()
        self._frame: int = 0

        # aktorzy do sprzątania
        self.actor_list: List[carla.Actor] = []
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None

        # do pieszych (żeby można było zatrzymać kontrolery)
        self.walker_controllers: List[carla.WalkerAIController] = []

        # kolejka obrazów z kamery
        self._image_queue: "queue.Queue[carla.Image]" = queue.Queue()

        # konfiguracja świata
        self._setup_world()

        # spawn ego + ruch uliczny + piesi
        self._spawn_vehicle()
        if self.num_traffic_vehicles > 0:
            self._spawn_traffic(self.num_traffic_vehicles)
        if self.num_pedestrians > 0:
            self._spawn_walkers(self.num_pedestrians)

    # ------------------------------------------------------------------ #
    # Konfiguracja świata
    # ------------------------------------------------------------------ #

    def _setup_world(self):
        """
        Ustawia tryb synchroniczny i docelowy FPS.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.fps)
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        # (opcjonalnie można tu dodać losową pogodę, ale na razie zostawiamy domyślną)

    def _restore_settings(self):
        """
        Przywraca oryginalne ustawienia świata i wyłącza synchroniczny TM.
        """
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self.traffic_manager = None

        self.world.apply_settings(self.original_settings)

    # ------------------------------------------------------------------ #
    # Spawn ego pojazdu + kamera
    # ------------------------------------------------------------------ #

    def _spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()

        # ego vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found.")
        spawn_point = np.random.choice(spawn_points)

        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            # próbujemy inne spawnpointy
            for sp in spawn_points:
                vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if vehicle is not None:
                    break
        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.vehicle = vehicle
        self.actor_list.append(vehicle)

        # Traffic Manager dla ego + NPC
        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(0)  # stały seed – stabilniej
        tm.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager = tm

        if self.autopilot:
            vehicle.set_autopilot(True, tm.get_port())
        else:
            vehicle.set_autopilot(False)

        # kamera RGB na masce
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_width))
        cam_bp.set_attribute("image_size_y", str(self.image_height))
        cam_bp.set_attribute("fov", str(self.fov))

        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=1.6),
            carla.Rotation(pitch=0.0),
        )
        camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        self.camera = camera
        self.actor_list.append(camera)

        # nowa kolejka na obrazy
        self._image_queue = queue.Queue()

        def _on_image(image: carla.Image):
            self._image_queue.put(image)

        camera.listen(_on_image)

        # kilka ticków na "rozgrzanie" sensora
        for _ in range(5):
            self.world.tick()

    # ------------------------------------------------------------------ #
    # NPC: pojazdy
    # ------------------------------------------------------------------ #

    def _spawn_traffic(self, num_vehicles: int = 5):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")

        spawn_points = self.world.get_map().get_spawn_points()
        np.random.shuffle(spawn_points)

        if self.traffic_manager is None:
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(0)
            tm.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager = tm
        else:
            tm = self.traffic_manager

        count = 0
        for sp in spawn_points:
            if count >= num_vehicles:
                break

            bp = np.random.choice(vehicle_bps)
            bp.set_attribute("role_name", "autopilot")

            try:
                veh = self.world.try_spawn_actor(bp, sp)
            except RuntimeError:
                veh = None

            if veh is None:
                continue

            veh.set_autopilot(True, tm.get_port())
            self.actor_list.append(veh)
            count += 1

        print(f"[CarlaEnv] Spawned {count} traffic vehicles.")

    # ------------------------------------------------------------------ #
    # NPC: piesi
    # ------------------------------------------------------------------ #

    def _spawn_walkers(self, num_walkers: int = 15):
        world = self.world
        blueprints = world.get_blueprint_library()

        walker_bps = blueprints.filter("walker.pedestrian.*")
        controller_bp = blueprints.find("controller.ai.walker")

        spawn_points: List[carla.Transform] = []
        for _ in range(num_walkers * 2):  # próbujemy trochę więcej punktów
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(loc))
            if len(spawn_points) >= num_walkers:
                break

        walkers: List[carla.Actor] = []
        controllers: List[carla.WalkerAIController] = []

        for sp in spawn_points:
            bp = np.random.choice(walker_bps)
            walker = world.try_spawn_actor(bp, sp)
            if walker is None:
                continue
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                continue

            walkers.append(walker)
            controllers.append(controller)
            self.actor_list.append(walker)
            self.actor_list.append(controller)

        world.tick()

        for controller in controllers:
            controller.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(1.0 + np.random.rand())  # 1.0–2.0 m/s

        self.walker_controllers = controllers
        print(f"[CarlaEnv] Spawned {len(walkers)} walkers.")

    # ------------------------------------------------------------------ #
    # Reset / Step / Close
    # ------------------------------------------------------------------ #

    def reset(self) -> np.ndarray:
        """
        Resetuje ego–pojazd + kamerę.
        NPC (inne auta + piesi) zostają w świecie.
        Zwraca pierwszą obserwację (H, W, 3) uint8.
        """
        # zabijamy starą kamerę + ego
        if self.camera is not None:
            self.camera.stop()
            try:
                if self.camera.is_alive:
                    self.camera.destroy()
            except RuntimeError:
                pass
            self.camera = None

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except RuntimeError:
                pass
            self.vehicle = None

        # ego od nowa
        self._spawn_vehicle()
        self._frame = 0

        obs = self._get_image(timeout=5.0)

        # ustaw spectator "jak w GTA"
        self._update_spectator()

        return obs

    def step(
        self,
        action: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Jeden krok symulacji.
        Jeśli autopilot=True, param action jest ignorowany.
        Zwraca (obs, reward, done, info).
        """
        if self.vehicle is None:
            raise RuntimeError("Call reset() before step().")

        # sterowanie ręczne tylko gdy autopilot=False
        if not self.autopilot:
            if action is None:
                raise ValueError("Action must be provided when autopilot=False.")
            throttle, steer = action
            throttle = float(np.clip(throttle, 0.0, 1.0))
            steer = float(np.clip(steer, -1.0, 1.0))
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0,
                hand_brake=False,
                reverse=False,
            )
            self.vehicle.apply_control(control)

        # tick świata
        self.world.tick()
        self._frame += 1

        # aktualizujemy spectator (widok GTA)
        self._update_spectator()

        obs = self._get_image(timeout=2.0)

        control = self.vehicle.get_control()
        action_vec = np.array(
            [control.throttle, control.steer, control.brake],
            dtype=np.float32,
        )

        reward = 0.0
        done = False
        info: Dict[str, Any] = {
            "frame": self._frame,
            "action": action_vec,
        }

        return obs, reward, done, info

    def close(self):
        """
        Niszczy wszystkie aktory i przywraca ustawienia świata.
        """
        if self.camera is not None:
            self.camera.stop()

        # zatrzymujemy kontrolery pieszych
        for ctrl in self.walker_controllers:
            try:
                ctrl.stop()
            except RuntimeError:
                pass
        self.walker_controllers = []

        for actor in self.actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass

        self.actor_list = []
        self.camera = None
        self.vehicle = None

        self._restore_settings()

    # ------------------------------------------------------------------ #
    # Pomocnicze: spectator + obraz
    # ------------------------------------------------------------------ #

    def _update_spectator(self):
        """
        Ustawia kamerę w oknie CARLI za autem (widok jak w GTA).
        """
        if self.vehicle is None:
            return
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(
                transform.location
                + transform.get_forward_vector() * -6.0
                + carla.Location(z=3.0),
                carla.Rotation(pitch=-10.0, yaw=transform.rotation.yaw),
            )
        )

    def _get_image(self, timeout: float = 1.0) -> np.ndarray:
        """
        Pobiera kolejną klatkę z kamery i zwraca (H, W, 3) uint8 w RGB.
        """
        try:
            image: carla.Image = self._image_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("Timeout while waiting for camera image.")

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        rgb = bgr[:, :, ::-1].copy()
        return rgb
