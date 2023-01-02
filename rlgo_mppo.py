import gym
import numpy as np

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from rlgo import GoEnv
from go import Go

def action_mask_fn(env):
    valid_pos = env.game.all_valid_positions(Go.BLACK)
    print(valid_pos)


def blah():
    e = GoEnv(board_shape=(9,9))
    #e.game.place_stone((0,0),Go.BLACK)
    action_mask_fn(e)

def train():
    def make_env():
        e = GoEnv(board_shape=(9,9))
        e = Monitor(e)
        e = ActionMasker(e, action_mask_fn)
        return e

    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env for i in range(num_cpu)])

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[128,128,128], vf=[128,128,128])])
    

    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn()

    model.learn(total_timesteps=int(1e6))
    model.save('mpgo-1')

if __name__ == "__main__": blah()
    