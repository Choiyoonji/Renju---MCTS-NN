import tensorflow.compat.v1 as tf
import numpy as np
import rule
tf.disable_v2_behavior() 

class CNN():
    """ Convolution Neural Network
    """
    def __init__(self, board_size, board = None):
        self.board_size = board_size
        self.board = board
        
        """ Input -> Common Layers -> Expert Crtic or Expert Policy
        """

        # Input
        self.input_states = tf.placeholder(tf.float32, [None, 4, board_size, board_size])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])

        # Common Layers
        self.conv1 = tf.nn.relu(tf.layers.conv2d(self.input_state, filters = 32, kernel_size = [3, 3], padding = "same"))
        self.conv2 = tf.nn.relu(tf.layers.conv2d(self.conv1, filters = 64, kernel_size = [3, 3], padding = "same"))
        self.conv3 = tf.nn.relu(tf.layers.conv2d(self.conv2, filters = 128, kernel_size = [3, 3], padding = "same"))

        # Expert Policy
        self.policy_conv = tf.nn.relu(tf.layers.conv2d(self.conv3, 1, [1, 1], padding = "same"))
        self.policy_conv_flat = tf.reshape(self.policy_conv, [-1, 1 * board_size * board_size])
        self.policy_fc = tf.nn.log_softmax(tf.layers.dense(self.policy_conv_flat, board_size * board_size))

        # Expert Critic
        self.critic_conv = tf.nn.relu(tf.layers.conv2d(self.conv3, 1, [1, 1], padding = "same"))
        self.critic_conv_flat = tf.reshape(self.policy_conv, [-1, 1 * board_size * board_size])
        self.critic_fc1 = tf.nn.relu(tf.layers.dense(self.critic_conv_flat, 64))
        self.critic_fc2 = tf.nn.tanh(tf.layers.dense(self.critic_fc1, 1))

        # Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.critic_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_size * board_size])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.policy_fc), 1)))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])

        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.policy_fc) * self.policy_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if board is not None:
            self.restore_model(board)

    def policy_value(self, state):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.policy_fc, self.critic_fc2],
                feed_dict={self.input_states: state}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board, stone, legal_positions):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        current_state = np.ascontiguousarray(board.reshape(
                -1, 4, self.board_size, self.board_size))
        act_probs, value = self.policy_value(current_state)

        legal_list = []

        for i in legal_positions:
            legal_list.append((i[0]-1)*15 + i[1])
        
        act_probs = zip(legal_positions, act_probs[0, legal_list])
        return act_probs, value

    def train_step(self, state, mcts_probs, winner, lr):
        """perform a training step"""
        winner = np.reshape(winner, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict = {self.input_states: state,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner,
                           self.learning_rate: lr})

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
