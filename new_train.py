"""
Alpha_Zero_Omok

@author: Capeelaa_K
"""
#future 모듈은 파이썬 버전 3.x면 필요X
import random
import numpy as np
import mcts2
import rule
from collections import defaultdict, deque
from CNN import CNN # Tensorflow

class Renju_Train():
    """
    Training Policy_value_net with MCTS guided Search
    """
    def __init__(self):
        """
        Define parameters of train, board and game
        """
        # Board
        self.board_size = 15
        self.n_in_row = 5
        self.board = [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        self.root_board = self.board
        self.root_board[7][7] = 1 
        
        # Train
        self.learn_rate = 2e-3
        self.n_playout = 400 # (num of simulations for each move)
        self.batch_size = 512 # mini-batch size
        self.epochs = 5 # repeat policy_update
        self.game_batch_num = 20
        self.c_puct = 1.0
        self.buffer_size = 100000
        self.data_buffer = deque(maxlen=self.buffer_size) # push play data

        # mini_batch
        self.batch_size = 512

        # Start Training with new policy_value_net
        self.policy_value_net = CNN(self.board_size)
        # Call MCTSplayer with policy_value_net
        self.mcts_player = mcts2.MCTS(self.board, self.board_size, self.policy_value_net.policy_value_fn, self.c_puct)
        # Define root_node
        self.root_node = mcts2.Node(1, state = self.root_board, board_size = self.board_size, parent_node = None)
        # model restore
        self.policy_value_net.restore_model('./current_policy.model')

    def get_equi_data(self, play_data):
        """
        Only for Go and Renju(Omok)
        Rotate and Flip Data

        play_data = [state, mcts_prob, winner] from game
        """

        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(mcts_prob, i)
                extend_data.append((equi_state,
                                    equi_mcts_prob.flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    equi_mcts_prob.flatten(),
                                    winner))
        return extend_data

    def collect_train_data(self):
        play_data = self.mcts_player.simulation(self.root_node) # ?
        play_data = list(play_data)

        self.episode_len = len(play_data) # length of play

        # Extend
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)

    def CNN_update(self):
        # mini_batch for effective training
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state = [data[0] for data in mini_batch]
        mcts_probs = [data[1] for data in mini_batch]
        winner = [data[2] for data in mini_batch]

        for _ in range(self.epochs):
            self.policy_value_net.train_step(
                state, mcts_probs, winner, self.learn_rate
            )
    
    def train(self):
        """
        Train the model
        """
        try:
            for i in range(self.game_batch_num):

                # play self game --> Collect play data 
                self.collect_train_data()

                print("game {}, episode len {}".format(i+1, self.episode_len))

                if len(self.data_buffer) > self.batch_size:
                    self.CNN_update()
                
                self.policy_value_net.save_model('./current_policy.model')
        
        except KeyboardInterrupt:
            print('\n\rquit')
        

TRAINING = Renju_Train()

TRAINING.train()
