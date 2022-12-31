import gym
import numpy as np
from stable_baselines3 import A2C, PPO

class OccupiedSpaceError(Exception): pass
class SelfCaptureError(Exception): pass
class KoError(Exception): pass

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
        self.board_history = [ self.board ]
        
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
       
    
    def check_ko(self, board):
        for previous_board in self.board_history:
            if np.array_equal(board, previous_board):
                return True

        return False

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
                self.place_stone(black_move, GoGame.BLACK)
            except (OccupiedSpaceError, SelfCaptureError) as e:
                info['black_legal'] = e.__class__.__name__
                return self.board, self.rewards['illegal'], False, info

        # play white
        for i in range(10):
            white_action = self.white_policy.predict(-self.board)[0]
            white_move, self.white_passes = self.parse_action(white_action)

            if self.white_passes:
                if black_passes:
                    return self.end_game(info)
                else:
                    break
            
            try:
                self.place_stone(white_move, GoGame.WHITE)
            except (OccupiedSpaceError, SelfCaptureError) as e:
                continue

            break

        info['white_passes'] = self.white_passes
        info['white_move'] = white_move

        return self.board, self.rewards['legal'], False, info

    def reset(self):
        self.board[:] = GoGame.EMPTY
        self.board_history = []
        self.white_passes = False
        return self.board

    def render(self, mode):
        print(self.board)

    def place_stone(self, pos, color):
        if self.board[pos[0],pos[1]] != GoGame.EMPTY:
            raise OccupiedSpaceError()

        board = np.copy(self.board)

        board[pos[0],pos[1]] = color
        
        positions = [            
            ( pos[0]-1, pos[1] ),
            ( pos[0]+1, pos[1] ),
            ( pos[0], pos[1]+1 ),
            ( pos[0], pos[1]-1 ),
            pos,
        ]

        for position in positions:
            group, group_color, liberties = self.get_group(position, board)

            if group and liberties == 0:
                if color == group_color: # self-capture
                    board[pos[0],pos[1]] = GoGame.EMPTY
                    raise SelfCaptureError
                else:
                    group = (
                        [v[0] for v in group],
                        [v[1] for v in group]
                    )
                    board[group] = GoGame.EMPTY
        
        if self.check_ko(board):
            raise KoError()        
        
        self.board_history.append(self.board)
        self.board = board
        

    def end_game(self, info):
        black_wins = (self.board == GoGame.BLACK).sum() > (self.board == GoGame.WHITE).sum()
        info['winner'] = 'black' if black_wins else 'white'
        print(f"{info['winner']} wins")
        reward = self.rewards['win'] if black_wins else self.rewards['loss']
        return self.board, reward, True, info 

    def get_group(self, pos, board):
        try:
            group_color = board[pos[0],pos[1]]
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

            if check_pos[0] < 0 or check_pos[0] >= board.shape[0] or check_pos[1] < 0 or check_pos[1] >= board.shape[1]:
                continue

            check_color = board[check_pos[0],check_pos[1]]

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

def train():
    env = GoGame(board_shape=(9,9))
    
    model = PPO('MlpPolicy', env, verbose=1)
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
    manual_test()
    #manual_eval()
    #test_ko()
    #train()