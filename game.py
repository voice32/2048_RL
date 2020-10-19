import numpy as np
import matplotlib.pyplot as plt

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

base_dir = '.'

class Game():
    """ 2048 game environment"""
    def __init__(self, size = 4, seed = 42, negative_reward = -10, reward_mode='log2', cell_move_penalty = 0.1):
        self.board_dim = size               # board dimension
        self.state_size = size * size       # total number of cells
        self.action_size = 4                # number of available actions
        np.random.seed(seed)
        self.best_game_history = []
        self.negative_reward = negative_reward
        self.reward_mode = reward_mode
        self.cell_move_penalty = cell_move_penalty

    def save_best_game_history(self):
        self.best_game_history = self.history.copy()
        with open(base_dir+'/best_game_hist.pkl', 'wb') as f:
            pickle.dump(self.best_game_history, f)
        
    def reset(self, init_fields = 2, step_penalty = 0, bootstrapping = False):
        """ Initializes the board
        
        Params
        ======
            init_fields (int): how many fields to fill initially
            step_penalty (int): the cost of an action
            bootstrapping (bool): whether to create a new (initial) board or simulate some intermediate game state
        """
        self.game_board = np.zeros((self.board_dim, self.board_dim))
        
        if not bootstrapping:
            for i in range(init_fields):
                self.fill_random_empty_cell()
        else:
            self.random_board()
            
        self.score = np.sum(self.game_board)
        self.reward = 0
        self.current_cell_move_penalty = 0
        self.done = False
        self.steps = 0
        self.rewards_list = []
        self.scores_list = []
        self.step_penalty = step_penalty
        self.history = []
        
        self.history.append({
            'action': -1,
            'new_board': self.game_board.copy(),  
            'old_board': None,
            'score': self.score,
            'reward': self.reward
        })
        
    def shift(self, board):
        """ Shifts all cells to the left and gathers penalties if needed """
        shifted_board = np.empty((board.shape[0], board.shape[1]))
        for i, row in enumerate(board):
            shifted = np.zeros(len(row))
            idx = 0
            for iv, v in enumerate(row):
                if v != 0:
                    shifted[idx] = v
                    if iv != idx:
                        self.current_cell_move_penalty += self.cell_move_penalty * v
                    idx += 1
            shifted_board[i] = shifted
        return shifted_board
        
    def calc_board(self, board):
        """ Calculate all cell mergers and return the new state of the board"""
        
        self.reward = 0
        self.current_cell_move_penalty = 0
        
        shifted_board = self.shift(board)
        
        merged_board = np.empty((shifted_board.shape[0], shifted_board.shape[1]))
        for idx, row in enumerate(shifted_board):
            for i in range(len(row)-1):
                if row[i] != 0 and row[i] == row[i+1]:
                    
                    row[i] = row[i] * 2
                    row[i+1] = 0
                    if self.reward_mode == 'log2':
                        self.reward += np.log2(row[i])
                    else:
                        self.reward += row[i]

            merged_board[idx] = row
        merged_board = self.shift(merged_board)
        
        return merged_board

    def current_state(self):
        """ Returns a flattened array of board cell values """
        return np.reshape(self.game_board.copy(), -1)
    
    def step(self, action, action_values):
        """ Applies the selected action to the board """
        old_board = self.game_board.copy()
        temp_board = self.game_board.copy()
        
        # Here we flip/transpose the board depending on the action in order to unify the calculation
        if action == ACTION_LEFT:
            temp_board = self.calc_board(temp_board)

        elif action == ACTION_RIGHT:
            temp_board = np.flip(self.calc_board(np.flip(temp_board, axis=1)), axis=1)

        elif action == ACTION_UP:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=0)), axis=0))

        elif action == ACTION_DOWN:
            temp_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(temp_board), axis=1)), axis=1))
        else: # just in case it happens
            return (self.game_board, 0, self.done)
        
        if not np.array_equal(self.game_board, temp_board):
            # Fill an empty cell with a new value
            self.game_board = temp_board.copy()
            self.fill_random_empty_cell()

            # Reward is the sum of the merged cells minus step cost
            self.reward = self.reward - self.current_cell_move_penalty
            
            self.score = np.sum(self.game_board)
            self.done = self.check_is_done()
            self.moved = True
        else:
            self.reward = self.negative_reward
            self.moved = False
        self.steps += 1
        self.rewards_list.append(self.reward)
        
        # Save the new state
        self.history.append({
            'action': action,
            'action_values': action_values,
            'old_board': old_board,
            'new_board': self.game_board.copy(),  
            'score': self.score,
            'reward': self.reward
        })

        return (self.game_board, self.reward, self.done)

    def virtual_step(self, action):
        if action == ACTION_LEFT:
            new_game_board = self.calc_board(self.game_board.copy())

        elif action == ACTION_RIGHT:
            new_game_board = np.flip(self.calc_board(np.flip(self.game_board, axis=1)), axis=1)

        elif action == ACTION_UP:
            new_game_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(self.game_board), axis=0)), axis=0))

        elif action == ACTION_DOWN:
            new_game_board = np.transpose(
                np.flip(
                    self.calc_board(np.flip(np.transpose(self.game_board), axis=1)), axis=1))
        else: # just in case it happens
            return (self.game_board, 0, self.done)
        
        self.reward = self.reward - self.step_penalty
        self.score = np.sum(self.game_board)
        self.done = self.check_is_done(new_game_board)
        return (new_game_board, self.reward, self.done)
    
    
    def check_is_done(self, board = None):
        """ Check if the game is over """
    
        if board is None:
            board = self.game_board
    
        # If there are at least one cell with 0, then the game is not over
        if not np.all(board):
            return False
        
        # If all cells are filled, we need to check if there are any possible moves
        else:
            # Check if there are any equal adjacent cells across horisontal and vertical axes
            for row in board:
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            for row in np.transpose(board):
                for cell in range(len(row) - 1):
                    if row[cell] == row[cell+1]:
                        return False
            
            # There are no equal adjacent cells, the game is over
            return True
    
    def print_board(self, transpose = False):
        """ Deprecated """
        if not transpose:
            print(self.game_board)
        else:
            print(np.transpose(self.game_board))
    
    def fill_random_empty_cell(self, playing=True):
        """ Finds an empty cell and fills it with 2 or 4 with 90/10% probability respectively (as per game rules on Wikipedia) """
        
        # If all cells are filled, there is no place to put a new value, just pass
        if np.all(self.game_board):
            return
        
        # Pick the cell
        x = np.random.randint(self.board_dim)
        y = np.random.randint(self.board_dim)
        
        # Check if it is empty, otherwise pick a new one
        while self.game_board[x, y] != 0:
            x = np.random.randint(self.board_dim)
            y = np.random.randint(self.board_dim)
        
        # If it is a regular game, only values 2 and 4 are allowed
        if playing:
            self.game_board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])
        else:
            # Otherwise it is a boostrapping game, then any values are allowed with certain probability
            self.game_board[x, y] = np.random.choice([2**i for i in range(1, 17)], p=np.linspace(1, 0.001, 16)/np.sum(np.linspace(1, 0.001, 16)))
        
    def draw_board(self, board = None, title = 'Current game'):
        """ Draws a colored game board """
        cell_colors = {
            0: '#FFFFFF',
            2: '#EEE4DA',
            4: '#ECE0C8',
            8: '#ECB280',
            16:'#EC8D53',
            32:'#F57C5F',
            64:'#E95937',
            128:'#F3D96B',
            256:'#F2D04A',
            512:'#E5BF2E',
            1024:'#E2B814',
            2048:'#EBC502',
            4096:'#00A2D8',
            8192:'#9ED682',
            16384:'#9ED682',
            32768:'#9ED682',
            65536:'#9ED682',
            131072:'#9ED682',
        }

        if board is None:
            board = self.game_board
        
        ncols = self.board_dim
        nrows = self.board_dim

        # create the plots
        fig = plt.figure(figsize=(3,3))
        plt.suptitle(title)
        axes = [ fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(0, nrows) for c in range(1, ncols+1) ]

        # add some data
        v = np.reshape(board, -1)
        for i, ax in enumerate(axes):
            ax.text(0.5, 0.5, str(int(v[i])), horizontalalignment='center', verticalalignment='center')
            ax.set_facecolor(cell_colors[int(v[i])])

        # remove the x and y ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
        
    def random_board(self):
        """ Creates a randomly filled board for bootstrapping """
        
        # Define how many cells we want to fill
        num_filled_cells = np.random.randint(12) + 4
        
        # Fill these cells
        for i in range(num_filled_cells):
            self.fill_random_empty_cell(playing=False)