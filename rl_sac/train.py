import time
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rl_sac.env import MazeCarEnv, make_env, EpisodeCounterCallback
from rl_sac.models import agent
from rl_sac.plotting import parse_urdf_and_plot
from rl_sac.env import MAZEFILE

num_envs = 1

RUN_NAME = f"sac_run_{time.strftime('%Y_%m_%d-%H_%M')}"
RUN_DIR = Path("runs") / RUN_NAME
TRAJECTORY_FILE = RUN_DIR / f"trajectories_{time.strftime('%Y_%m_%d-%H_%M')}.npy"
TRAJECTORY_PLOT = RUN_DIR / f"trajectories_{time.strftime('%Y_%m_%d-%H_%M')}.png"
TRAJECTORY_EVAL_FILE = (
    RUN_DIR / f"trajectories_eval_{time.strftime('%Y_%m_%d-%H_%M')}.npy"
)
TRAJECTORY_EVAL_PLOT = RUN_DIR / f"trajectories_eval_{time.strftime('%Y_%m_%d-%H_%M')}.png"

TENSORBOARD_DIR = RUN_DIR / 'tensorboard'
MODEL_DIR = RUN_DIR / 'models'
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Initial environment check
env = MazeCarEnv(render_mode="human", inference_agent=agent)
try:
    check_env(env)
    env.close()
except Exception as e:
    env.close()
    raise

env = SubprocVecEnv([make_env(render_mode=None) for _ in range(num_envs)])
env = VecMonitor(env, str(TENSORBOARD_DIR))

model = SAC(
    "MultiInputPolicy",
    env,
    ent_coef=0.15,
    gamma=0.97,
    tensorboard_log=str(TENSORBOARD_DIR),
    batch_size=256,
    learning_rate=3e-4,
    train_freq=(512, "step"),
    gradient_steps=512,
    buffer_size=200_000,
    device="cuda",
)

TOTAL_EPISODES = 5
RESUME_TRAINING = False

max_steps = env.get_attr('max_steps_per_episode')[0]
total_timesteps = TOTAL_EPISODES * max_steps

counter_callback = EpisodeCounterCallback()

if RESUME_TRAINING and (MODEL_DIR / 'sac_model.zip').exists():
    model = SAC.load(str(MODEL_DIR / 'sac_model'), env=env)
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        tb_log_name=RUN_NAME,
        progress_bar=True,
        callback=counter_callback,
    )
else:
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=RUN_NAME,
        progress_bar=True,
        callback=counter_callback,
    )

model.save(str(MODEL_DIR / 'sac_model'))

total_epochs_trained = counter_callback.episode_count
print(f"\nTotal episodes completed: {total_epochs_trained}")

# Save training trajectories and plot
all_trajs = env.get_attr("all_trajectories")
all_trajs_flattened = [traj for env_trajs in all_trajs for traj in env_trajs]
np.save(str(TRAJECTORY_FILE), np.array(all_trajs_flattened, dtype=object))
parse_urdf_and_plot(
    MAZEFILE, all_trajs_flattened, TRAJECTORY_PLOT, total_epochs_trained, trainingrun=True
)

env.close()

# --- Evaluation ---
custom_goal_pos = [-3.0, 0.0]

eval_env = MazeCarEnv(render_mode="human", inference_agent=agent)
model = SAC.load(str(MODEL_DIR/"sac_model"), env=eval_env)

p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=-0.6, cameraPitch=-85, cameraTargetPosition=[0,0,0])

eval_env.start_pos = [2.0, -1.0, 0.1]
num_eval_epochs = 4
max_steps_per_epoch = eval_env.max_steps_per_episode

all_eval_trajectories = []
for epoch in range(num_eval_epochs):
    print(f"Epoch {epoch+1}/{num_eval_epochs}")
    obs, info = eval_env.reset(custom_goal_pos=None)
    cumulated_reward = 0.0
    trajectory = []
    for step in range(max_steps_per_epoch):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        time.sleep(1./480.0)
        cumulated_reward += reward
        car_pos, _ = p.getBasePositionAndOrientation(eval_env.carId, physicsClientId=eval_env.client)
        trajectory.append(car_pos[:2])
        if terminated or truncated:
            break
    all_eval_trajectories.append(trajectory)
    print(f"Epoch {epoch+1} finished with Cumulated Reward: {cumulated_reward}")

np.save(str(TRAJECTORY_EVAL_FILE), np.array(all_eval_trajectories, dtype=object))
parse_urdf_and_plot(
    MAZEFILE,
    all_eval_trajectories,
    TRAJECTORY_EVAL_PLOT,
    total_epochs_trained,
    num_eval_epochs,
    trainingrun=False,
)

eval_env.close()
