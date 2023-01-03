import gym
import numpy as np
import torch as th

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from rlgo import GoEnv
from go import Go

def action_mask_fn(env):
    valid_pos = env.game.all_valid_positions(Go.BLACK)
    valid_actions = set(
        p[0]*env.game.board_shape[0] + p[1]
        for p in valid_pos
    )
    valid_actions.add(np.prod(env.num_actions-1))
    return [ i in valid_actions for i in range(env.num_actions) ]

def train():
    def make_env():
        e = GoEnv(board_shape=(9,9))
        e = Monitor(e)
        e = ActionMasker(e, action_mask_fn)
        return e

    num_cpu = 4  # Number of processes to use
    #env = SubprocVecEnv([make_env for i in range(num_cpu)])
    env = make_env()

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[128,128,128], vf=[128,128,128])])
    

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=int(1e5))
    model.save('mpgo-1')

    env.set_white_policy(MaskablePPO.load('mpgo-1'))
    model.learn(total_timesteps=int(1e5))
    model.save('mpgo-2')

    env.set_white_policy(MaskablePPO.load('mpgo-2'))
    model.learn(total_timesteps=int(1e5))
    model.save('mpgo-3')

if __name__ == "__main__": train()
    