import pybullet as p
from rl_sac.models import agent

from stable_baselines3.common.callbacks import BaseCallback

import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from rl_sac.models import beta

MAZEFILE = "urdf/maze_colored.urdf"

class MazeCarEnv(gym.Env):
    metadata = {'render_modes': ['human'], "render_fps": 500}

    def __init__(self, render_mode=None, inference_agent=None):
        super().__init__()

        # --- Define Action Space ---
        # Continuous actions: [left_motor_speed, right_motor_speed] ∈ [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # --- Define Observation Space ---
        # Beobachtung: [Auto_x, Auto_y, Auto_yaw, Ziel_x, Ziel_y]
        low = np.array([-6, -6, -np.pi, -6, -6], dtype=np.float32)
        high = np.array([6, 6, np.pi, 6, 6], dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=np.array([-6, -6, -np.pi, -6, -6], dtype=np.float32),
                high=np.array([6, 6, np.pi, 6, 6], dtype=np.float32),
                dtype=np.float32
            ),
            "camera": spaces.Box(
                low=0,
                high=255,
                shape=(64, 64, 4),  # RGBA Bild vom PyBullet
                dtype=np.uint8
            )
        })

        # --- PyBullet Setup ---
        self.render_mode = render_mode
        self.client = p.connect(p.DIRECT if render_mode is None else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)  # Standard 240 Hz

        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.mazeId = p.loadURDF(
            MAZEFILE,
            basePosition=[0, 0, 0],
            useFixedBase=True,  # keeps maze static
            physicsClientId=self.client,
        )

        self.goal_area_1 = np.array([-5.5, 4.5])   # oberer Ausgang
        self.goal_area_2 = np.array([5.5, -4.5])  # unterer Ausgang
        self.goal_radius = 0.5
        self.target_goal_pos = None  # Wird in reset() gesetzt
        self.correct_goal_index = None  # Zufällige Wahl des Ziels

        self.goal_spheres = []

        # Roboter (Auto) laden
        self.start_pos = [-1.0, -2.0, 0.1] 
        self.start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.carId = p.loadURDF("urdf/simple_two_wheel_car.urdf",
                                self.start_pos, self.start_orn,
                                physicsClientId=self.client)

        self.left_wheel_joint_index = 1
        self.right_wheel_joint_index = 0

        self.step_counter = 0
        self.max_steps_per_episode = 480

        self.action_repeat = 50

        self.trajectory = []
        self.all_trajectories = []

        self.was_in_goal = False 
        self.stopped = False

        self.agent = inference_agent

    def _get_obs(self):
        pos, orn_quat = p.getBasePositionAndOrientation(self.carId, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(orn_quat)
        yaw = euler[2]

        camera_image = self._get_camera_image()

        return {
            "state": np.array([pos[0], pos[1], yaw, self.target_goal_pos[0], self.target_goal_pos[1]], dtype=np.float32),
            "camera": camera_image
        }

    def _get_info(self):
        car_pos, _ = p.getBasePositionAndOrientation(self.carId, physicsClientId=self.client)
        dist_goal1 = np.linalg.norm(np.array(car_pos[:2]) - self.goal_area_1)
        dist_goal2 = np.linalg.norm(np.array(car_pos[:2]) - self.goal_area_2)

        return {
            "distance_goal1": dist_goal1,
            "distance_goal2": dist_goal2,
            "target_goal_index": self.correct_goal_index
        }

    def reset(self, seed=None, options=None, custom_goal_pos=None):
        super().reset(seed=seed)
        self.step_counter = 0

        # Auto zurücksetzen
        start_x = self.start_pos[0] + self.np_random.uniform(-0.3, 0.3)
        start_y = self.start_pos[1] + self.np_random.uniform(-0.3, 0.3)
        start_yaw = self.np_random.uniform(-np.pi/6, np.pi/6)
        start_orn = p.getQuaternionFromEuler([0, 0, start_yaw])
        p.resetBasePositionAndOrientation(self.carId, [start_x, start_y, self.start_pos[2]], start_orn, physicsClientId=self.client)
        p.resetBaseVelocity(self.carId,
                            linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0],
                            physicsClientId=self.client)

        # Entferne bestehende Ziel-Sphären
        for sphere_id in self.goal_spheres:
            p.removeBody(sphere_id, physicsClientId=self.client)
        self.goal_spheres.clear()

        goal_visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.goal_radius,
            rgbaColor=[0, 1, 0, 0.8]  # Bright green
        )

        # Set the goal position
        if custom_goal_pos is not None:
            self.target_goal_pos = np.array(custom_goal_pos)
            self.correct_goal_index = -1  # Custom goal, no correct index
        else:
            self.correct_goal_index = self.np_random.integers(0, 2)
            self.target_goal_pos = self.goal_area_1 if self.correct_goal_index == 0 else self.goal_area_2

        # if self.correct_goal_index == 0:
        #     self.target_goal_pos = self.goal_area_1

        #     if self.render_mode == "human":
        #         goal_id = p.createMultiBody(
        #             baseVisualShapeIndex=goal_visual_shape,
        #             basePosition=[self.goal_area_1[0], self.goal_area_1[1], 0.1],
        #             useMaximalCoordinates=True
        #         )
        #         self.goal_spheres.append(goal_id)
        # else:
        #     self.target_goal_pos = self.goal_area_2

        #     if self.render_mode == "human":
        #         goal_id = p.createMultiBody(
        #             baseVisualShapeIndex=goal_visual_shape,
        #             basePosition=[self.goal_area_2[0], self.goal_area_2[1], 0.1],
        #             useMaximalCoordinates=True
        #         )
        #         self.goal_spheres.append(goal_id)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        car_pos, _ = p.getBasePositionAndOrientation(self.carId, physicsClientId=self.client)
        self.prev_dist_to_goal = np.linalg.norm(np.array(car_pos[:2]) - self.target_goal_pos)

        if len(self.trajectory) > 0:
            self.all_trajectories.append(self.trajectory)  # Save old episode

        self.trajectory = []
        self.trajectory.append(car_pos[:2])  # only x, y

        self.was_in_goal = False  
        self.stopped = False

        return observation, info

    def step(self, action):

        # Clip the action to make sure it's in [-1, 1] range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Scale action to desired motor speeds
        max_motor_speed = 10.0  # you can adjust this if needed
        left_vel = float(action[0] * max_motor_speed)
        right_vel = float(action[1] * max_motor_speed)

        # left_vel += np.random.normal(0, 0.2)
        # right_vel += np.random.normal(0, 0.2)

        p.setJointMotorControl2(
            bodyUniqueId=self.carId,
            jointIndex=self.left_wheel_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=20.0
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.carId,
            jointIndex=self.right_wheel_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=20.0
        )

        # Jetzt die Simulation mehrmals updaten,
        for _ in range(self.action_repeat):
            p.stepSimulation()

        self.step_counter += 1

        # Beobachtung + Reward + Done bestimmen
        observation = self._get_obs()
        info = self._get_info()

        terminated = False

        reward = 0.0
        reward -= 0.005

        # Calculate distance to the goal
        car_pos, _ = p.getBasePositionAndOrientation(self.carId, physicsClientId=self.client)
        curr_dist_to_goal = np.linalg.norm(np.array(car_pos[:2]) - self.target_goal_pos)

        # Reward for getting closer to the goal
        reward = 1 * (self.prev_dist_to_goal - curr_dist_to_goal)  # Positive reward for reducing distance
        self.prev_dist_to_goal = curr_dist_to_goal

        # Check if the car is in the goal area
        in_goal_1 = np.linalg.norm(car_pos[:2] - self.goal_area_1) < self.goal_radius
        in_goal_2 = np.linalg.norm(car_pos[:2] - self.goal_area_2) < self.goal_radius
        in_goal = np.linalg.norm(car_pos[:2] - self.target_goal_pos) < self.goal_radius

        # Check if the car is stationary in the goal area
        linear_velocity, angular_velocity = p.getBaseVelocity(self.carId, physicsClientId=self.client)
        speed = np.linalg.norm(linear_velocity)  # Calculate the speed from linear velocity

        if in_goal_1:
            if self.correct_goal_index == 0:
                if not self.was_in_goal:
                    reward += 5.0
                self.was_in_goal = True
                if speed < 0.05 and self.stopped == False:  # Threshold to consider the car stationary
                    reward = +3.0
                    self.stopped = True
                    terminated = True
            else:
                reward += -5.0
                terminated = True
                self.was_in_goal = True
        elif in_goal_2:
            if self.correct_goal_index == 1:
                if not self.was_in_goal:
                    reward += 5.0
                self.was_in_goal = True
                if speed < 0.05 and self.stopped == False:  # Threshold to consider the car stationary
                    reward = +3.0
                    self.stopped = True
                    terminated = True
            else:
                reward += -5.0
                terminated = True
                self.was_in_goal = True
        elif in_goal:
            if self.correct_goal_index == -1:
                if not self.was_in_goal:
                    reward += 5.0
                self.was_in_goal = True
                if speed < 0.05 and self.stopped == False:  # Threshold to consider the car stationary
                    reward = +3.0
                    self.stopped = True
                    terminated = True
                else:
                    reward += -5.0
                    terminated = True
                    self.was_in_goal = True


########################################################
        # === Intrinsische Motivation / Free Energy Reward hinzufügen ===
        current_state = observation["state"]
        camera_image = observation["camera"]
        next_state = observation["state"][:3]  # x, y, yaw

        # Vorhersage-Fehler (Free Energy)
        predicted = self.agent.predict_next_state(current_state, camera_image)
        free_energy = self.agent.compute_prediction_error(next_state, predicted)

        # Store for visualization
        self.last_current_state = current_state
        self.last_camera_image = camera_image
        self.last_predicted_state = predicted
        self.last_actual_next_state = next_state

        # Totaler Reward: extrinsisch + intrinsisch
        reward -= beta * free_energy
        
        

        self.agent.update_model(current_state, camera_image, next_state)

        self.trajectory.append(car_pos[:2])

        # print(self.prev_dist_to_goal)

        contacts = p.getContactPoints(bodyA=self.carId, bodyB=self.mazeId, physicsClientId=self.client)
        if len(contacts) > 0:
            reward -= 3.0
            terminated = True 

        truncated = (self.step_counter >= self.max_steps_per_episode)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        # Bei PyBullet im GUI-Modus passiert das Rendering automatisch
        pass

    def _render_frame(self):
        if not hasattr(self, 'last_camera_image'):
            return
        plt.figure(1, figsize=(3,3))
        plt.clf()
        plt.imshow(self.last_camera_image)
        plt.axis('off')
        title = (
            f'state: {np.round(self.last_current_state,2)}\n'
            f'pred: {np.round(self.last_predicted_state,2)}\n'
            f'act:  {np.round(self.last_actual_next_state,2)}'
        )
        plt.title(title)
        plt.pause(0.001)

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def _get_camera_image(self):
        car_pos, car_orn = p.getBasePositionAndOrientation(self.carId, physicsClientId=self.client)
        car_euler = p.getEulerFromQuaternion(car_orn)

        camera_distance = 0.1
        camera_height = 0.2
        camera_yaw = car_euler[2] * 180 / np.pi
        target_pos = [
            car_pos[0] + camera_distance * np.cos(car_euler[2]),
            car_pos[1] + camera_distance * np.sin(car_euler[2]),
            car_pos[2] + camera_height
        ]

        width, height, rgba, _, _ = p.getCameraImage(
            width=64,
            height=64,
            viewMatrix=p.computeViewMatrix(
                cameraEyePosition=[car_pos[0], car_pos[1], car_pos[2] + camera_height],
                cameraTargetPosition=target_pos,
                cameraUpVector=[0, 0, 1]
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=70,
                aspect=1.0,
                nearVal=0.01,
                farVal=10.0
            ),
            physicsClientId=self.client
        )

        rgba_image = np.reshape(rgba, (height, width, 4)).astype(np.uint8)
        return rgba_image

class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
        return True

def make_env(render_mode=None):
    def _init():
        return MazeCarEnv(render_mode=render_mode, inference_agent=agent)
    return _init
