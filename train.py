from gru_package.gru_ppo import GruPPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import make_vec_env
import random, os
import numpy as np 
import torch as th 

from tensorboardX import SummaryWriter
from gymnasium.wrappers.record_video import RecordVideo 
random.seed(0)
np.random.seed(0)
th.manual_seed(0)
if th.cuda.is_available():
    th.cuda.manual_seed_all(0)
    th.cuda.manual_seed(0)

LOG_DIR = '/home/cai/Desktop/GRU_AC/runs'
ENV_NAME = "CartPole-v1"
BATCH_SIZE = 32
N_STEPS = 64
LR = 3e-4
TOTAL_STEP = 10_000
RENDER = False 
RENDER_MODE = 'rgb_array'
TEST_EPI = 10 
RECODE_TEST_EP = True 

env = make_vec_env(env_id = "CartPole-v1", n_envs = 8 )
recurrent_model = RecurrentPPO("MlpLstmPolicy", 
                               env, 
                               verbose=1,
                               tensorboard_log= os.path.join(LOG_DIR, 'LSTM_PPO')
                               )
recurrent_model.learn(TOTAL_STEP)
writer_1 = SummaryWriter(os.path.join(LOG_DIR, 'LSTM_PPO'))
del env 

env = make_vec_env(env_id = "CartPole-v1", n_envs = 8)
gru_model = GruPPO("MlpGruPolicy", 
                   env, 
                   verbose=1,
                   tensorboard_log= os.path.join(LOG_DIR, 'GRU_PPO')
                   )
gru_model.learn(TOTAL_STEP)
writer_2 = SummaryWriter(os.path.join(LOG_DIR, 'GRU_PPO'))

del env


model = PPO("MlpPolicy", 
            "CartPole-v1",
                n_steps = 128,
                batch_size=128 , 
                verbose=1,
                tensorboard_log=os.path.join(LOG_DIR, 'PPO'))
model.learn(TOTAL_STEP)


writer_3 = SummaryWriter(os.path.join(LOG_DIR, 'PPO'))

import gymnasium as gym 


agents_rewards = []
agents_epi_lenghts = []
for index ,(model, writer) in enumerate(zip([recurrent_model,gru_model, model],[writer_1, writer_2, writer_3])):
    reward_list = []
    epi_lenght_list = []
    env = gym.make(ENV_NAME, 
        render_mode = RENDER_MODE,
        )

    env = RecordVideo(env= env, 
                      video_folder= os.path.join(LOG_DIR, ENV_NAME, 'video'), 
                      name_prefix= 'LSTM' if index == 0 else "GRU" if index == 1 else 'PPO' 
                        ) if RECODE_TEST_EP else env
    total_reward = 0
    step = 0
    for epi in range(TEST_EPI):
        is_done = False
        observation, _ = env.reset()
        reward_ = 0
        epi_lenght = 0

        while not is_done:
            action, _= model.policy.predict(observation, deterministic = True) 
            observation, reward, terminated, truncated, _ = env.step(action)
            epi_lenght += 1
            reward_ += reward
            if RENDER:
                env.render()
            if terminated or truncated:
                is_done = True 
            writer.add_scalar('eval/cummulated_reward',total_reward, step)
            step += 1 
            total_reward += reward

        reward_list.append(reward_)
        epi_lenght_list.append(epi_lenght)
    del env
    agents_rewards.append([np.mean(reward_list), np.var(reward_list), np.std(reward_list)])
    agents_epi_lenghts.append([np.mean(epi_lenght_list), np.var(epi_lenght_list), np.std(epi_lenght_list)])

print(f'[ENV_NAME]: {ENV_NAME}, [TEST_EPI]: {TEST_EPI}')

print(f'[LSTM]\n\
      [reward mean]: {agents_rewards[0][0]}, [reward var]: {agents_rewards[0][1]}, [reward std]: {agents_rewards[0][2]}\n\
        [episode mean]: {agents_epi_lenghts[0][0]}, [episode var]: {agents_epi_lenghts[0][1]}, [episode std]: {agents_epi_lenghts[0][2]}')

print(f'[GRU]\n\
      [reward mean]: {agents_rewards[1][0]}, [reward var]: {agents_rewards[1][1]}, [reward std]: {agents_rewards[1][2]}\n\
        [episode mean]: {agents_epi_lenghts[1][0]}, [episode var]: {agents_epi_lenghts[1][1]}, [episode std]: {agents_epi_lenghts[1][2]}')


print(f'[PPO]\n\
      [reward mean]: {agents_rewards[2][0]}, [reward var]: {agents_rewards[2][1]}, [reward std]: {agents_rewards[2][2]}\n\
        [episode mean]: {agents_epi_lenghts[2][0]}, [episode var]: {agents_epi_lenghts[2][1]}, [episode std]: {agents_epi_lenghts[2][2]}')