import pygame
import random
import time
from collections import defaultdict
from snake_game import SnakeGame, Direction, GameState

class RandomAgent:
    def __init__(self):
        self.name = "Random Agent"
    
    def get_valid_moves(self, snake, current_direction, grid_size):
        """Get all valid moves that don't cause immediate collision"""
        head_x, head_y = snake[0]
        valid_moves = []
        
        # Define all possible directions and their restrictions
        directions = [
            (Direction.UP, Direction.DOWN),
            (Direction.DOWN, Direction.UP),
            (Direction.LEFT, Direction.RIGHT),
            (Direction.RIGHT, Direction.LEFT)
        ]
        
        for direction, opposite in directions:
            # Skip if it's the opposite of current direction (can't reverse)
            if direction == opposite and current_direction == opposite:
                continue
            if current_direction == direction and direction == opposite:
                continue
            if current_direction.name == opposite.name:
                continue
                
            # Calculate new position
            dx, dy = direction.value
            new_x, new_y = head_x + dx, head_y + dy
            
            # Check if move is valid (no wall collision, no self collision)
            if (0 <= new_x < grid_size and 0 <= new_y < grid_size and 
                (new_x, new_y) not in snake):
                valid_moves.append(direction)
        
        return valid_moves
    
    def choose_action(self, game_info):
        """Choose a random valid action"""
        snake = game_info['snake']
        current_direction = None
        
        # We need to get current direction from the game state
        # For now, we'll determine it from snake movement
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
            current_direction = Direction.RIGHT  # Default
        
        valid_moves = self.get_valid_moves(snake, current_direction, game_info['grid_size'])
        
        if valid_moves:
            return random.choice(valid_moves)
        else:
            # No valid moves - return current direction (will cause game over)
            return current_direction

class GameRunner:
    def __init__(self, show_gui=False, game_speed=10):
        self.show_gui = show_gui
        self.game_speed = game_speed
        self.stats = defaultdict(list)
    
    def run_single_game(self, agent, max_steps=1000):
        """Run a single game with the given agent"""
        game = SnakeGame()
        if not self.show_gui:
            # Disable pygame display for faster training
            pygame.display.set_mode((1, 1))
        
        step_count = 0
        
        while game.game_state == GameState.PLAYING and step_count < max_steps:
            if self.show_gui:
                # Handle pygame events to prevent window hanging
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return game.score, step_count, False  # User quit
            
            # Get agent's action
            game_info = game.get_game_info()
            action = agent.choose_action(game_info)
            
            # Apply action to game
            game.next_direction = action
            game.update()
            
            if self.show_gui:
                game.draw()
                pygame.time.wait(int(1000 / self.game_speed))
            
            step_count += 1
        
        return game.score, step_count, True
    
    def run_multiple_games(self, agent, num_games=100, max_steps=1000):
        """Run multiple games and collect statistics"""
        scores = []
        steps = []
        
        print(f"Running {num_games} games with {agent.name}...")
        
        for i in range(num_games):
            if i % 10 == 0:
                print(f"Game {i+1}/{num_games}")
            
            score, step_count, completed = self.run_single_game(agent, max_steps)
            if completed:
                scores.append(score)
                steps.append(step_count)
        
        # Calculate statistics
        if scores:
            stats = {
                'mean_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'mean_steps': sum(steps) / len(steps),
                'max_steps': max(steps),
                'min_steps': min(steps),
                'games_completed': len(scores),
                'total_games': num_games
            }
        else:
            stats = {
                'mean_score': 0, 'max_score': 0, 'min_score': 0,
                'mean_steps': 0, 'max_steps': 0, 'min_steps': 0,
                'games_completed': 0, 'total_games': num_games
            }
        
        return stats, scores, steps
    
    def print_statistics(self, stats, agent_name):
        """Print formatted statistics"""
        print(f"\n=== {agent_name} Performance Statistics ===")
        print(f"Games completed: {stats['games_completed']}/{stats['total_games']}")
        print(f"Average score: {stats['mean_score']:.2f}")
        print(f"Best score: {stats['max_score']}")
        print(f"Worst score: {stats['min_score']}")
        print(f"Average steps per game: {stats['mean_steps']:.2f}")
        print(f"Longest game: {stats['max_steps']} steps")
        print(f"Shortest game: {stats['min_steps']} steps")

def main():
    print("=== Phase 2: Random AI Baseline ===")
    
    # Create random agent
    agent = RandomAgent()
    
    # Option to watch a single game
    print("\nOptions:")
    print("1. Watch single game with GUI")
    print("2. Run performance evaluation (100 games, no GUI)")
    print("3. Run quick test (10 games, no GUI)")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Single game with GUI
        runner = GameRunner(show_gui=True, game_speed=10)
        print("Running single game with GUI...")
        score, steps, completed = runner.run_single_game(agent, max_steps=1000)
        if completed:
            print(f"Game completed! Score: {score}, Steps: {steps}")
        else:
            print("Game was interrupted.")
    
    elif choice == "2":
        # Full performance evaluation
        runner = GameRunner(show_gui=False)
        stats, scores, steps = runner.run_multiple_games(agent, num_games=100, max_steps=1000)
        runner.print_statistics(stats, agent.name)
        
        # Save results for later comparison
        print(f"\nScore distribution: {sorted(scores)}")
        print(f"Games with score > 0: {len([s for s in scores if s > 0])}")
        print(f"Games with score >= 5: {len([s for s in scores if s >= 5])}")
        print(f"Games with score >= 10: {len([s for s in scores if s >= 10])}")
    
    elif choice == "3":
        # Quick test
        runner = GameRunner(show_gui=False)
        stats, scores, steps = runner.run_multiple_games(agent, num_games=10, max_steps=1000)
        runner.print_statistics(stats, agent.name)
    
    else:
        print("Invalid choice. Running quick test...")
        runner = GameRunner(show_gui=False)
        stats, scores, steps = runner.run_multiple_games(agent, num_games=10, max_steps=1000)
        runner.print_statistics(stats, agent.name)

if __name__ == "__main__":
    main()