import gym
import numpy as np
from stable_baselines3 import A2C, PPO

class GoGame(gym.Env):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, board_shape=None):
        super().__init__()

        self.board_shape = board_shape if board_shape else (9,9)

        self.white_passes = False
        self.num_actions = np.prod(self.board_shape)+1
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=self.board_shape, 
            dtype=np.int8)

        self.action_space = gym.spaces.Discrete(n=self.num_actions)

        self.white_policy = self

        self.rewards = dict(
            loss=-1,
            illegal=-.1,
            legal=0,
            win=1,
        )

    def predict(self, obs):
        return self.action_space.sample()

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
                self.place_stone(black_move, GoGame.BLACK)
            except IndexError:
                info['black_legal'] = False
                return self.board, self.rewards['illegal'], False, info

        # play white
        for i in range(10):
            white_action = self.white_policy.predict(-self.board)
            white_move, self.white_passes = self.parse_action(white_action)

            if self.white_passes:
                if black_passes:
                    return self.end_game(info)
                else:
                    break
            
            try:
                self.place_stone(white_move, GoGame.WHITE)
            except IndexError:
                continue

            break

        info['white_passes'] = self.white_passes
        info['white_move'] = white_move

        return self.board, 1, False, info

    def reset(self):
        self.board[:] = GoGame.EMPTY
        self.white_passes = False
        return self.board

    def render(self, mode):
        print(self.board)

    def place_stone(self, pos, color):
        self.board[pos[0],pos[1]] = color
        
        positions = [
            pos,
            ( pos[0]-1, pos[1] ),
            ( pos[0]+1, pos[1] ),
            ( pos[0], pos[1]+1 ),
            ( pos[0], pos[1]-1 )
        ]

        capture = False
        for position in positions:
            group, color, liberties = self.get_group(position)

            if group and liberties == 0:
                group = (
                    [v[0] for v in group],
                    [v[1] for v in group]
                )
                self.board[group] = GoGame.EMPTY
                capture = True

    def end_game(self, info):
        black_wins = (self.board == GoGame.BLACK).sum() > (self.board == GoGame.WHITE).sum()
        info['winner'] = 'black' if black_wins else 'white'
        reward = self.rewards['win'] if black_wins else self.rewards['loss']
        return self.board, reward, True, info 

    def get_group(self, pos):
        
        try:
            group_color = self.board[pos[0],pos[1]]
        except IndexError:
            return None, None, None
        
        if group_color == GoGame.EMPTY:
            return None, None, None

        liberties = 0
        group = [ ]
        to_check = set()
        visited = set()

        to_check.add(pos)

        while len(to_check):
            check_pos = to_check.pop()

            if check_pos in visited:
                continue

            visited.add(check_pos)

            if check_pos[0] < 0 or check_pos[0] >= self.board.shape[0] or check_pos[1] < 0 or check_pos[1] >= self.board.shape[1]:
                continue

            check_color = self.board[check_pos[0],check_pos[1]]

            if check_color == group_color:
                group.append(check_pos)

                to_check.update([
                    ( check_pos[0]+1, check_pos[1] ),
                    ( check_pos[0]-1, check_pos[1] ),
                    ( check_pos[0], check_pos[1]+1 ),
                    ( check_pos[0], check_pos[1]-1 )
                ])
            elif check_color == GoGame.EMPTY:
                liberties += 1

        return group, group_color, liberties

    def parse_action(self, action):
        if action == self.num_actions-1:
            return None, True
        else:
            return ( 
                int(action / self.board_shape[0]),
                int(action % self.board_shape[0])
                ), False  

def main():
    env = GoGame()
    
    model = PPO('MlpPolicy', env, verbose=1).learn(total_timesteps=100000)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
        if dones[0]:
            break

if __name__ == "__main__": main()