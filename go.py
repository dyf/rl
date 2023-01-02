import numpy as np

class OccupiedSpaceError(Exception): pass
class SelfCaptureError(Exception): pass
class KoError(Exception): pass

def neighbors(pos, board):
    return [ p for p in [
        ( pos[0]-1, pos[1] ),
        ( pos[0]+1, pos[1] ),
        ( pos[0], pos[1]+1 ),
        ( pos[0], pos[1]-1 )
    ]  if p[0] >= 0 and p[1] >= 0 and p[0] < board.shape[0] and p[1] < board.shape[1] ]

class StoneGroup:
    def __init__(self, color):
        self.color = color
        self.positions = set()
        self.liberties = set()

    def add_stone(self, pos, board):
        self.positions.add(pos)
        self.liberties.discard(pos)

        for n in neighbors(pos, board):            
            if board[n[0],n[1]] == Go.EMPTY:
                self.liberties.add(n)
    
    def remove_liberty(self, pos):
        self.liberties.discard(pos)

    def add_liberty(self, pos, board):        
        for p in self.positions:
            if pos in neighbors(p, board):
                self.liberties.add(pos)

    def copy(self):
        g = StoneGroup(color=self.color)
        g.positions = self.positions.copy()
        g.liberties = self.liberties.copy()
        return g

    def __str__(self):
        return f"color:{self.color} pos:{self.positions} lib:{self.liberties}"
    
    @staticmethod
    def merge(groups):        
        assert len(set(g.color for g in groups)) == 1

        new_group = StoneGroup(groups[0].color)

        for g in groups:
            new_group.positions.update(g.positions)
            new_group.liberties.update(g.liberties)

        new_group.liberties -= new_group.positions
        
        return new_group

class Go:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def __init__(self, board_shape=(19,19)):
        self.board_shape = board_shape
        self.stone_groups = []

        self.white_passes = False
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        self.board_history = [ self.board ]


    def is_occupied(self, pos):
        return self.board[pos[0],pos[1]] != Go.EMPTY


    def is_ko(self, board):
        for previous_board in self.board_history:
            if np.array_equal(board, previous_board):
                return True

        return False

    def all_valid_positions(self, color):
        it = np.nditer(self.board, flags=['multi_index'])
        valid = set()

        for x in it:
            pos = (it.multi_index[0], it.multi_index[1])

            try:                
                self.place_stone(pos, color, commit=False)
                valid.add(pos)
            except (OccupiedSpaceError, SelfCaptureError, KoError) as e:
                continue
        
        return valid 

    def place_stone(self, pos, color, commit=True):

        if self.is_occupied(pos):
            raise OccupiedSpaceError()
                
        board = np.copy(self.board)

        board[pos[0],pos[1]] = color

        all_groups = []
        merge_groups = []

        for g in self.stone_groups:
            g = g.copy()
            if g.color == color and pos in g.liberties: # adding stone to existing group
                g.add_stone(pos, board)
                merge_groups.append(g)
            else: # removing liberty from existing group
                g.remove_liberty(pos)
                all_groups.append(g)

        if len(merge_groups) == 0:
            g = StoneGroup(color=color)
            g.add_stone(pos, board)
            all_groups.append(g)
        else:        
            all_groups.append(StoneGroup.merge(merge_groups))

        # check for capture
        remaining_groups = []
        for g in all_groups:
            if len(g.liberties) == 0:
                if pos in g.positions:
                    raise SelfCaptureError()

                x = (
                    [v[0] for v in g.positions],
                    [v[1] for v in g.positions]
                )
                board[x] = Go.EMPTY                
                for og in all_groups:
                    if og is not g:
                        for p in g.positions:
                            og.add_liberty(p, board)
            else:
                remaining_groups.append(g)
        
        if self.is_ko(board):
            raise KoError()

        if commit:
            self.stone_groups = remaining_groups
            self.board = board
            self.board_history.append(self.board)

def test_merge():
    g = Go(board_shape=(5,5))
    g.place_stone((0,0), Go.BLACK)
    g.place_stone((1,0), Go.BLACK)
    g.place_stone((0,2), Go.BLACK)
    g.place_stone((1,2), Go.BLACK)
    g.place_stone((0,1), Go.BLACK)
    print(g.board)
    for x in g.stone_groups:
        print(x)

def test_capture():
    g = Go(board_shape=(5,5))
    g.place_stone((0,0), Go.BLACK)
    print(g.board)
    g.place_stone((1,0), Go.WHITE)
    print(g.board)
    g.place_stone((0,1), Go.WHITE)
    print(g.board)

def test_ko():
    g = Go(board_shape=(5,5))

    g.place_stone((1,1), Go.BLACK)
    g.place_stone((2,0), Go.BLACK)
    g.place_stone((3,1), Go.BLACK)
    g.place_stone((2,2), Go.BLACK)
    g.place_stone((1,2), Go.WHITE)
    g.place_stone((2,3), Go.WHITE)
    g.place_stone((3,2), Go.WHITE)
    print(g.board)
    g.place_stone((2,1), Go.WHITE)
    print(g.board)
    g.place_stone((2,2), Go.BLACK)
    print(g.board)

def test_self_capture():
    g = Go(board_shape=(5,5))

    g.place_stone((1,1), Go.BLACK)
    g.place_stone((2,0), Go.BLACK)
    g.place_stone((3,1), Go.BLACK)
    g.place_stone((2,2), Go.BLACK)
    print(g.board)
    g.place_stone((2,1), Go.WHITE)
    print(g.board)    

def test_occupied():
    g = Go(board_shape=(5,5))

    g.place_stone((1,1), Go.BLACK)
    g.place_stone((1,1), Go.WHITE)

def test_valid():
    g = Go(board_shape=(3,3))
    g.place_stone((0,0), Go.BLACK)    
    print("checking")
    print(g.all_valid_positions(Go.BLACK))

if __name__ == "__main__":
    test_valid()