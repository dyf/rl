import gym
import numpy as np
from go import Go, KoError, OccupiedSpaceError, SelfCaptureError
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch as th

class GoGame(gym.Env):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, board_shape=None):
        super().__init__()

        self.game = Go(board_shape if board_shape else (9,9))

        self.white_passes = False
        self.num_actions = np.prod(self.game.board_shape)+1
        
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=self.game.board_shape, 
            dtype=np.int8)

        self.action_space = gym.spaces.Discrete(n=self.num_actions)

        self.white_policy = self

        self.rewards = dict(
            loss=-100,
            illegal=-.1,
            legal=.1,
            win=100,
        )
       
    def predict(self, obs):
        return self.action_space.sample(), None

    def step(self, action):
        black_move, black_passes = self.parse_action(action)
        info = {
            'black_move': black_move,
            'black_passes': black_passes,
            'black_legal': True
        }

        # play black
        if black_passes:
            if self.white_passes:
                return self.end_game(info)
        else:
            try:
                self.game.place_stone(black_move, Go.BLACK)
            except (OccupiedSpaceError, SelfCaptureError, KoError) as e:
                info['black_legal'] = e.__class__.__name__
                return self.game.board, self.rewards['illegal'], False, info

        # play white 
        num_white_tries = 1000
        for i in range(num_white_tries):
            white_action = self.white_policy.predict(-self.game.board)[0]
            white_move, self.white_passes = self.parse_action(white_action)

            if self.white_passes:
                if black_passes:
                    return self.end_game(info)
                else:
                    break
            
            try:
                self.game.place_stone(white_move, Go.WHITE)                
            except (OccupiedSpaceError, SelfCaptureError, KoError) as e:
                continue

            break

        info['white_passes'] = self.white_passes
        info['white_move'] = white_move

        return self.game.board, self.rewards['legal'], False, info

    def reset(self):
        self.game = Go(board_shape=self.game.board_shape)
        self.white_passes = False
        return self.game.board

    def render(self, mode):
        print(self.game.board)

    
        

    def end_game(self, info):
        black_wins = (self.game.board == Go.BLACK).sum() > (self.game.board == Go.WHITE).sum()
        info['winner'] = 'black' if black_wins else 'white'
        print(f"{info['winner']} wins")
        reward = self.rewards['win'] if black_wins else self.rewards['loss']
        return self.game.board, reward, True, info 

    

    def parse_action(self, action):
        if action == self.num_actions-1:
            return None, True
        else:
            return ( 
                int(action / self.game.board_shape[0]),
                int(action % self.game.board_shape[0])
                ), False  

def train():
    def make_env():
        return Monitor(GoGame(board_shape=(9,9)))

    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env for i in range(num_cpu)])

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[128,128,128], vf=[128,128,128])])
    
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=int(1e6))
    model.save('go-1')
    
    env.white_policy = PPO.load('go-1')
    model.learn(total_timesteps=int(1e6))
    model.save('go-2')

    env.white_policy = PPO.load('go-2')
    model.learn(total_timesteps=int(1e6))
    model.save('go-3')


    

def manual_test():
    env = GoGame(board_shape=(5,5))
    for i in range(100):
        a = env.action_space.sample()
        _,_,_,info = env.step(a)
        print(env.board)
        print(info)
        input()

def manual_eval():
    env = GoGame(board_shape=(9,9))
    model = PPO.load('go-2')

    for i in range(100):
        a = model.predict(env.board)[0]
        _,_,_,info = env.step(a)
        print(env.board)
        print(info)
        input()

def test_ko():
    env = GoGame(board_shape=(5,5))
    env.board = np.array([
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0],
        [ 1,  0,  1, -1,  0],
        [ 0,  1, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
    ])

    env.render(mode=1)

    env.place_stone((2,1), GoGame.WHITE)
    env.render(mode=1)
    env.place_stone((2,2), GoGame.BLACK)
    env.render(mode=1)
    
if __name__ == "__main__": 
    #manual_test()
    #manual_eval()
    #test_ko()
    train()