import numpy as np
import random
import pickle
import os
from collections import defaultdict, deque
from snake_game import SnakeGame, Direction, GameState
from random_agent import GameRunner

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.name = "Q-Learning Agent"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training statistics
        self.training_scores = []
        self.training_episodes = []
        self.epsilon_history = []
        
    def get_food_direction(self, head_pos, food_pos):
        """Get direction to food (8-directional)"""
        head_x, head_y = head_pos
        food_x, food_y = food_pos
        
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Normalize to direction
        if dx > 0 and dy == 0: return "E"
        elif dx < 0 and dy == 0: return "W"
        elif dx == 0 and dy > 0: return "S"
        elif dx == 0 and dy < 0: return "N"
        elif dx > 0 and dy > 0: return "SE"
        elif dx > 0 and dy < 0: return "NE"
        elif dx < 0 and dy > 0: return "SW"
        elif dx < 0 and dy < 0: return "NW"
        else: return "SAME"
    
    def get_danger_state(self, snake, current_direction, grid_size):
        """Get danger state: left danger, straight danger, right danger"""
        head_x, head_y = snake[0]
        
        # Direction mappings for relative directions
        if current_direction == Direction.UP:
            left_dir = Direction.LEFT
            straight_dir = Direction.UP
            right_dir = Direction.RIGHT
        elif current_direction == Direction.DOWN:
            left_dir = Direction.RIGHT
            straight_dir = Direction.DOWN
            right_dir = Direction.LEFT
        elif current_direction == Direction.LEFT:
            left_dir = Direction.DOWN
            straight_dir = Direction.LEFT
            right_dir = Direction.UP
        else:  # RIGHT
            left_dir = Direction.UP
            straight_dir = Direction.RIGHT
            right_dir = Direction.DOWN
        
        dangers = []
        for direction in [left_dir, straight_dir, right_dir]:
            dx, dy = direction.value
            new_x, new_y = head_x + dx, head_y + dy
            
            # Check if position is dangerous
            is_danger = (new_x < 0 or new_x >= grid_size or 
                        new_y < 0 or new_y >= grid_size or 
                        (new_x, new_y) in snake)
            dangers.append(int(is_danger))
        
        return tuple(dangers)
    
    def get_state(self, game_info, current_direction):
        """Convert game info to state representation"""
        snake = game_info['snake']
        food = game_info['food']
        grid_size = game_info['grid_size']
        
        head_pos = snake[0]
        
        # Food direction (8-directional)
        food_dir = self.get_food_direction(head_pos, food)
        
        # Danger state (left, straight, right relative to current direction)
        danger_state = self.get_danger_state(snake, current_direction, grid_size)
        
        # Distance to walls in current direction
        head_x, head_y = head_pos
        dx, dy = current_direction.value
        
        wall_distance = 0
        test_x, test_y = head_x, head_y
        while (0 <= test_x + dx < grid_size and 0 <= test_y + dy < grid_size):
            test_x += dx
            test_y += dy
            wall_distance += 1
        
        # Discretize wall distance to reduce state space
        wall_distance = min(wall_distance, 10)  # Cap at 10
        
        return (food_dir, danger_state, wall_distance, current_direction.name)
    
    def get_valid_actions(self, current_direction):
        """Get valid actions (can't reverse direction)"""
        if current_direction == Direction.UP:
            return [Direction.UP, Direction.LEFT, Direction.RIGHT]
        elif current_direction == Direction.DOWN:
            return [Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        elif current_direction == Direction.LEFT:
            return [Direction.LEFT, Direction.UP, Direction.DOWN]
        else:  # RIGHT
            return [Direction.RIGHT, Direction.UP, Direction.DOWN]
    
    def choose_action(self, game_info, current_direction=None, training=True):
        """Choose action using epsilon-greedy policy"""
        if current_direction is None:
            # Determine current direction from snake movement
            snake = game_info['snake']
            if len(snake) >= 2:
                head = snake[0]
                neck = snake[1]
                dx = head[0] - neck[0]
                dy = head[1] - neck[1]
                
                if dx == 1: current_direction = Direction.RIGHT
                elif dx == -1: current_direction = Direction.LEFT
                elif dy == 1: current_direction = Direction.DOWN
                elif dy == -1: current_direction = Direction.UP
            else:
                current_direction = Direction.RIGHT
        
        state = self.get_state(game_info, current_direction)
        valid_actions = self.get_valid_actions(current_direction)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose best action based on Q-values
            q_values = {action: self.q_table[state][action] for action in valid_actions}
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        """Update Q-value using Bellman equation"""
        current_q = self.q_table[state][action]
        
        # Find max Q-value for next state
        if next_valid_actions:
            max_next_q = max(self.q_table[next_state][next_action] for next_action in next_valid_actions)
        else:
            max_next_q = 0
        
        # Bellman equation
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_reward(self, game_info, previous_game_info, action_taken):
        """Calculate reward for the action taken"""
        current_score = game_info['score']
        previous_score = previous_game_info['score'] if previous_game_info else 0
        current_state = game_info['game_state']
        
        # Food reward
        if current_score > previous_score:
            return 10
        
        # Death penalty
        if current_state == GameState.GAME_OVER:
            return -10
        
        # Step penalty (encourage efficiency)
        return -0.1
    
    def train_episode(self, max_steps=1000):
        """Train for one episode"""
        game = SnakeGame()
        # Disable pygame display for faster training
        import pygame
        pygame.display.set_mode((1, 1))
        
        step_count = 0
        previous_game_info = None
        previous_state = None
        previous_action = None
        
        while game.game_state == GameState.PLAYING and step_count < max_steps:
            # Get current game info
            current_game_info = game.get_game_info()
            
            # Determine current direction
            snake = current_game_info['snake']
            current_direction = Direction.RIGHT  # Default
            if len(snake) >= 2:
                head = snake[0]
                neck = snake[1]
                dx = head[0] - neck[0]
                dy = head[1] - neck[1]
                
                if dx == 1: current_direction = Direction.RIGHT
                elif dx == -1: current_direction = Direction.LEFT
                elif dy == 1: current_direction = Direction.DOWN
                elif dy == -1: current_direction = Direction.UP
            
            # Choose action
            action = self.choose_action(current_game_info, current_direction, training=True)
            
            # Apply action
            game.next_direction = action
            game.update()
            
            # Get new game info after action
            new_game_info = game.get_game_info()
            
            # Update Q-value for previous action
            if previous_game_info is not None:
                reward = self.get_reward(current_game_info, previous_game_info, previous_action)
                current_state = self.get_state(current_game_info, current_direction)
                next_valid_actions = self.get_valid_actions(current_direction)
                
                self.update_q_value(previous_state, previous_action, reward, current_state, next_valid_actions)
            
            # Store for next iteration
            previous_game_info = current_game_info.copy()
            previous_state = self.get_state(current_game_info, current_direction)
            previous_action = action
            
            step_count += 1
        
        # Handle final state (game over or max steps reached)
        if previous_game_info is not None:
            final_game_info = game.get_game_info()
            reward = self.get_reward(final_game_info, previous_game_info, previous_action)
            self.update_q_value(previous_state, previous_action, reward, None, [])
        
        return game.score, step_count
    
    def train(self, episodes=1000, save_interval=100):
        """Train the Q-learning agent"""
        print(f"Training Q-learning agent for {episodes} episodes...")
        
        for episode in range(episodes):
            if episode % save_interval == 0:
                print(f"Episode {episode}/{episodes}, Epsilon: {self.epsilon:.3f}")
            
            # Train one episode
            score, steps = self.train_episode()
            
            # Store training statistics
            self.training_scores.append(score)
            self.training_episodes.append(episode)
            self.epsilon_history.append(self.epsilon)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Print progress
            if episode % save_interval == 0 and episode > 0:
                recent_scores = self.training_scores[-save_interval:]
                avg_score = sum(recent_scores) / len(recent_scores)
                print(f"Average score (last {save_interval} episodes): {avg_score:.2f}")
        
        print("Training completed!")
        return self.training_scores, self.epsilon_history
    
    def evaluate(self, num_games=100):
        """Evaluate the trained agent"""
        print(f"Evaluating trained agent over {num_games} games...")
        
        # Temporarily disable exploration
        old_epsilon = self.epsilon
        self.epsilon = 0  # Pure exploitation
        
        runner = GameRunner(show_gui=False)
        stats, scores, steps = runner.run_multiple_games(self, num_games=num_games)
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        return stats, scores, steps
    
    def save(self, filename):
        """Save the trained model"""
        data = {
            'q_table': dict(self.q_table),
            'training_scores': self.training_scores,
            'training_episodes': self.training_episodes,
            'epsilon_history': self.epsilon_history,
            'epsilon': self.epsilon
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        self.training_scores = data['training_scores']
        self.training_episodes = data['training_episodes']
        self.epsilon_history = data['epsilon_history']
        self.epsilon = data['epsilon']
        print(f"Model loaded from {filename}")

def main():
    print("=== Phase 3: Q-Learning Agent ===")
    
    agent = QLearningAgent()
    
    print("\nOptions:")
    print("1. Train new agent (1000 episodes)")
    print("2. Train new agent (5000 episodes)")
    print("3. Load and evaluate existing model")
    print("4. Quick training test (100 episodes)")
    
    choice = input("Choose option (1/2/3/4): ").strip()
    
    if choice == "1":
        # Train for 1000 episodes
        scores, epsilon_history = agent.train(episodes=1000)
        agent.save("snake_qlearning_1000.pkl")
        
        # Evaluate
        stats, scores, steps = agent.evaluate(num_games=100)
        print("\n=== Evaluation Results ===")
        print(f"Average score: {stats['mean_score']:.2f}")
        print(f"Best score: {stats['max_score']}")
        print(f"Games with score > 0: {len([s for s in scores if s > 0])}")
        print(f"Games with score >= 5: {len([s for s in scores if s >= 5])}")
        print(f"Games with score >= 10: {len([s for s in scores if s >= 10])}")
        
    elif choice == "2":
        # Train for 5000 episodes
        scores, epsilon_history = agent.train(episodes=5000)
        agent.save("snake_qlearning_5000.pkl")
        
        # Evaluate
        stats, scores, steps = agent.evaluate(num_games=100)
        print("\n=== Evaluation Results ===")
        print(f"Average score: {stats['mean_score']:.2f}")
        print(f"Best score: {stats['max_score']}")
        print(f"Games with score > 0: {len([s for s in scores if s > 0])}")
        print(f"Games with score >= 5: {len([s for s in scores if s >= 5])}")
        print(f"Games with score >= 10: {len([s for s in scores if s >= 10])}")
        
    elif choice == "3":
        # Load existing model
        filename = input("Enter model filename: ").strip()
        if os.path.exists(filename):
            agent.load(filename)
            stats, scores, steps = agent.evaluate(num_games=100)
            print("\n=== Evaluation Results ===")
            print(f"Average score: {stats['mean_score']:.2f}")
            print(f"Best score: {stats['max_score']}")
        else:
            print("File not found!")
    
    elif choice == "4":
        # Quick training test
        scores, epsilon_history = agent.train(episodes=100)
        stats, scores, steps = agent.evaluate(num_games=20)
        print("\n=== Quick Test Results ===")
        print(f"Average score: {stats['mean_score']:.2f}")
        print(f"Best score: {stats['max_score']}")
    
    else:
        print("Invalid choice. Running quick test...")
        scores, epsilon_history = agent.train(episodes=100)
        stats, scores, steps = agent.evaluate(num_games=20)
        print("\n=== Quick Test Results ===")
        print(f"Average score: {stats['mean_score']:.2f}")
        print(f"Best score: {stats['max_score']}")

if __name__ == "__main__":
    main()