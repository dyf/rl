import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C

class GoGame(gym.Env):
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    
    def __init__(self, color):
        self.color = color
        self.board_shape = (9,9)
        self.passed = False
        self.opponent_passed = False
        self.num_actions = np.prod(self.board_shape)+1
        self.board = np.zeros(self.board_shape, dtype=int)
        self.opponent_passed = False

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.board_shape, dtype=int)

        self.action_space = gym.spaces.Discrete(n=self.num_actions)

        self.rewards = dict(
            loss=-100,
            illegal=-.1,
            legal=0.1,
            win=100,
        )

    def reset(self):
        self.board[:] = 0
        self.opponent_passed = False
        self.passed = False

    def step(self, action):
        move, is_pass = self.parse_action(action)
        print(f"{self.color} to {move}")
        
        self.passed = is_pass

        # game over
        if is_pass and self.opponent_passed:
            won = (self.board == self.BLACK).sum() > (self.board == self.WHITE).sum()
            status = 'win' if won else 'loss'
            reward = self.rewards[status] if won else self.rewards[status]
            return self.board, reward, True, False, status 

        # check for illegal move: already a piece there
        if self.board[move[0],move[1]] != self.EMPTY:
            return self.board, self.rewards['illegal'], False, False, "illegal move"

        # place piece
        self.board[move[0],move[1]] = self.color

        positions = [
            move,
            ( move[0]-1, move[1] ),
            ( move[0]+1, move[1] ),
            ( move[0], move[1]+1 ),
            ( move[0], move[1]-1 )
        ]

        capture = False
        for position in positions:
            group, color, liberties = self.get_group(position)

            if group and liberties == 0:
                group = (
                    [v[0] for v in group],
                    [v[1] for v in group]
                )
                self.board[group] = self.EMPTY
                capture = True
                
                

        return self.board, self.rewards['legal'], False, False, "capture" if capture else "legal move"


    def get_group(self, pos):
        if pos[0] < 0 or pos[0] >= self.board.shape[0] or pos[1] < 0 or pos[1] >= self.board.shape[1]:
            return None, None, None 
        
        group_color = self.board[pos[0],pos[1]]
        
        if group_color == self.EMPTY:
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
            elif check_color == self.EMPTY:
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
        
class CustomCallback(BaseCallback):
    def __init__(self, opponent_env, verbose=0):
        self.opponent_env = opponent_env
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        np.copyto(self.opponent_env.board, self.training_env.board)
        self.opponent_env.opponent_passed = self.training_env.passed
        a = self.opponent_env.action_space.sample()
        r = e_black.step(a)
        
        np.copyto(self.training_env.board, self.opponent_env.board)
        self.training_env.opponent_passed = self.training_env.passed
        return True

def main_train():
    black_env = GoGame(color=GoGame.BLACK)
    white_env = GoGame(color=GoGame.WHITE)
    callback = CustomCallback(opponent_env=white_env)

    model = A2C('MlpPolicy', black_env).learn(total_timesteps=1000, callback=callback)

def main_test():
    e = GoGame(color=GoGame.WHITE)
    e.board = np.array([
        [ 1, 1, -1, 0, 0],
        [ 0, -1, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0],
    ])
    print(e.board)
    e.step(5)
    print(e.board)

def main_compete():
    e_black = GoGame(color=GoGame.BLACK)
    e_white = GoGame(color=GoGame.WHITE)
    e_white.reset()
    e_black.reset()
    is_pass = False

    for i in range(5):
        np.copyto(e_black.board, e_white.board)
        e_black.opponent_passed = is_pass
        a = e_black.action_space.sample()
        _, is_pass = e_black.parse_action(a)
        r = e_black.step(a)
        print(e_black.board, r[4])

        input("next")
        np.copyto(e_white.board, e_black.board)
        e_white.opponent_passed = is_pass
        a = e_white.action_space.sample()
        _, is_pass = e_white.parse_action(a)
        r = e_white.step(a)
        print(e_white.board, r[4])

        input("next")
    
if __name__ == "__main__": main_train()