import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv

import pandas as pd

train_df = pd.read_csv('./datasets/bot_train_ETHBTC_700_hour.csv')
train_df = train_df.sort_values('Date')

test_df = pd.read_csv('./datasets/bot_rollout_ETHBTC_700_hour.csv')
test_df = test_df.sort_values('Date')

train_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(train_df, serial=True)])

model = PPO2(MlpPolicy, train_env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=5000)

test_env = DummyVecEnv(
    [lambda: BitcoinTradingEnv(test_df, serial=True)])

obs = test_env.reset()
for i in range(50000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render(mode="human", title="BTC")

test_env.close()
