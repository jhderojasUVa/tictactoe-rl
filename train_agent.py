import numpy as np
import random
import json
from collections import defaultdict
import time

# --- 1. GAME LOGIC IMPLEMENTATION ---

# MAPPING: X=1, O=-1, EMPTY=0
# Board is represented as a flat 9-element numpy array for easy hashing.
X_PLAYER = 1
O_PLAYER = -1
EMPTY = 0

class TicTacToeGame:
    """Encapsulates the Tic-Tac-Toe game environment."""
    def __init__(self):
        # 3x3 board stored as a flat 1D array of 9 integers
        self.board = np.full(9, EMPTY, dtype=int)
        self.current_player = X_PLAYER
        self.is_game_over = False

    def reset(self):
        """Resets the board for a new game."""
        self.board = np.full(9, EMPTY, dtype=int)
        self.current_player = X_PLAYER
        self.is_game_over = False
        return self.get_state_hash()

    def get_available_actions(self):
        """Returns a list of indices (0-8) where moves can be made."""
        return np.where(self.board == EMPTY)[0].tolist()

    def make_move(self, action):
        """
        Applies a move and returns the reward and whether the game is over.
        :param action: Index (0-8) where the current_player places their mark.
        :return: (reward, game_over)
        """
        if self.board[action] != EMPTY:
            # Should not happen during training, but handles invalid move
            return -100, True # Penalty for illegal move
        
        self.board[action] = self.current_player
        
        reward = 0
        game_over = self.check_win(self.current_player)

        if game_over:
            reward = 100 # High reward for winning
        elif len(self.get_available_actions()) == 0:
            game_over = True
            reward = 10 # Small reward for a draw (better than a loss)
        
        if game_over:
            self.is_game_over = True
        else:
            # Switch player for the next turn
            self.current_player = O_PLAYER if self.current_player == X_PLAYER else X_PLAYER
            
        return reward, game_over

    def check_win(self, player):
        """Checks if the given player has won."""
        board = self.board
        
        # Define all 8 winning combinations (indices in the 1D array)
        winning_combos = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
            (0, 4, 8), (2, 4, 6)             # Diagonals
        ]

        for a, b, c in winning_combos:
            if board[a] == board[b] == board[c] == player:
                return True
        return False

    def get_state_hash(self):
        """
        Generates a unique string hash of the current board state.
        This is used as the key in the Q-Table dictionary.
        Encoding: 0=Empty, 1=X, 2=O.
        """
        # Convert -1 (O) to 2 for easier hashing and consistency
        # Board: [1, 0, -1, 0, 1, 0, -1, 0, 1] -> [1, 0, 2, 0, 1, 0, 2, 0, 1]
        temp_board = np.where(self.board == O_PLAYER, 2, self.board)
        return "".join(map(str, temp_board))

# --- 2. Q-LEARNING AGENT ---

class QLearningAgent:
    """Reinforcement learning agent using Q-learning for move selection and policy updates."""
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.99995):
        # Q-Table structure: {state_hash: [Q_value_for_action_0, ..., Q_value_for_action_8]}
        # We use defaultdict to initialize unseen states with a default list of zeros
        # Since the value is a list (mutable), we must initialize it properly on first access
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate (starts high)
        self.epsilon_decay = epsilon_decay # Rate at which epsilon shrinks
        self.player_symbol = X_PLAYER # Agent always plays as X for training purposes

    def get_q_values(self, state_hash):
        """Returns the Q-values array for a given state, initializing it if new."""
        if state_hash not in self.q_table:
            # Initialize with 0s for the 9 possible actions
            self.q_table[state_hash] = np.full(9, 0.0, dtype=float)
        return self.q_table[state_hash]

    def choose_action(self, game, is_training):
        """
        Selects an action using the epsilon-greedy strategy.
        :param game: The current TicTacToeGame instance.
        :param is_training: If True, uses epsilon-greedy; otherwise, uses pure exploitation (best move).
        :return: The chosen action index (0-8).
        """
        available_actions = game.get_available_actions()
        current_state_q_values = self.get_q_values(game.get_state_hash())

        if is_training and random.random() < self.epsilon:
            # Exploration: Choose a random valid move
            return random.choice(available_actions)
        else:
            # Exploitation: Choose the move with the highest Q-value
            # Set invalid moves to a very low score so they are never picked
            q_values_for_valid_moves = current_state_q_values.copy()
            
            # Find all invalid moves (those not in available_actions)
            all_actions = set(range(9))
            invalid_actions = list(all_actions - set(available_actions))
            
            # Penalize invalid actions
            q_values_for_valid_moves[invalid_actions] = -1e9 # Effectively rule out occupied cells

            # Select the action index with the maximum Q-value
            best_action = np.argmax(q_values_for_valid_moves)
            return best_action

    def update_q_value(self, prev_state_hash, action, reward, next_state_hash):
        """Applies the Q-Learning update rule."""
        
        # Q(s, a)
        old_q = self.get_q_values(prev_state_hash)[action]
        
        # max_{a'} Q(s', a')
        if next_state_hash is None:
            # Terminal state (game ended), no future Q-value
            max_future_q = 0
        else:
            # Get the max Q-value from the next state (s')
            max_future_q = np.max(self.get_q_values(next_state_hash))

        # Q-Learning Formula Update: Q(s, a) <- Q(s, a) + alpha * [R + gamma * max_future_q - Q(s, a)]
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        
        # Apply the update
        self.q_table[prev_state_hash][action] = new_q

    def save_q_table(self, filename="q_table.json"):
        """
        Saves the trained Q-Table to a JSON file.
        The Q-values (numpy array) must be converted to a list for JSON serialization.
        """
        serializable_table = {
            state: q_values.tolist() for state, q_values in self.q_table.items()
        }
        with open(filename, 'w') as f:
            json.dump(serializable_table, f, indent=4)
        print(f"\n✅ Q-Table saved successfully to {filename}. Size: {len(self.q_table)} states.")


# --- 3. TRAINING FUNCTION ---

def train(episodes=50000):
    """Main training loop."""
    game = TicTacToeGame()
    agent = QLearningAgent()
    
    # Simple agent that makes random moves for the opponent (O)
    def random_move(game):
        actions = game.get_available_actions()
        return random.choice(actions)

    print(f"Starting Q-Learning training for {episodes} episodes...")
    start_time = time.time()
    
    # Stats tracking
    stats = {'wins': 0, 'losses': 0, 'draws': 0}

    for episode in range(1, episodes + 1):
        # Reset game and get initial state
        state = game.reset()
        game_history = [] # To store (state, action, reward, next_state) tuples
        
        # --- Game Play Loop ---
        while not game.is_game_over:
            
            player_to_move = game.current_player
            
            if player_to_move == X_PLAYER:
                # 1. X_PLAYER (RL Agent) chooses an action
                action = agent.choose_action(game, is_training=True)
                
                # Store the state and action before the move is made
                # We will update the reward and next_state later
                game_history.append({'state': state, 'action': action, 'reward': 0, 'next_state': None})

            else:
                # 2. O_PLAYER (Random Opponent) moves
                action = random_move(game)
                # The agent (X) does not track O's moves in its policy/Q-table
            
            # 3. Apply move and get result
            reward, game_over = game.make_move(action)
            next_state = game.get_state_hash() if not game_over else None

            # --- Q-Table Update ---
            # Update the *last* recorded move in the game history, if it was an RL Agent move (X_PLAYER)
            if player_to_move == X_PLAYER:
                
                # Check if the move resulted in a terminal state
                if next_state is None:
                    # If game is over (win/loss/draw), use the final reward
                    
                    # If X won (player is O because game.make_move switched player before checking win)
                    if game.check_win(X_PLAYER):
                        final_reward = 100
                        stats['wins'] += 1
                    # If O won (player is X)
                    elif game.check_win(O_PLAYER):
                        final_reward = -100
                        stats['losses'] += 1
                    # Draw
                    else:
                        final_reward = 10
                        stats['draws'] += 1
                    
                    # Backpropagate the final reward through the agent's move history
                    # We process the history in reverse (last move first)
                    for i in reversed(range(len(game_history))):
                        history_entry = game_history[i]
                        
                        # Use the final reward for the very last move (R)
                        # And use the updated Q-value of the next state (which is 0 for terminal)
                        agent.update_q_value(
                            prev_state_hash=history_entry['state'], 
                            action=history_entry['action'], 
                            reward=final_reward, 
                            next_state_hash=next_state
                        )
                        # The next state for the previous move will be the current state's max Q
                        # (This is a simplification for a low-depth game like TicTacToe)
                        next_state = history_entry['state']
                        final_reward = 0 # Future moves receive instant reward of 0

                state = next_state # Prepare for the next loop iteration (if game is not over)
            
            # Decay epsilon (exploration rate)
            agent.epsilon *= agent.epsilon_decay
            # Ensure a minimum level of exploration (e.g., 1%)
            agent.epsilon = max(0.01, agent.epsilon)

        # Print progress every 5000 episodes
        if episode % 5000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{episodes} | Win Rate (X): {(stats['wins'] / episode) * 100:.2f}% | Epsilon: {agent.epsilon:.4f} | Time: {elapsed_time:.2f}s")
            
    # Final save and cleanup
    agent.save_q_table()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training complete. Total time: {total_time:.2f} seconds.")


# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    """Entry point for training the Tic-Tac-Toe RL agent."""
    train(episodes=150000) # You can adjust the number of training episodes here