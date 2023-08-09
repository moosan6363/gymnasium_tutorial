from gymnasium.wrappers import TimeLimit, RecordVideo
import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

if __name__ == '__main__':
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = RecordVideo(env, "./video")
    obs = env.reset()
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=12800)
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
