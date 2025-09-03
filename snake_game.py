import pygame
import random
import sys
from enum import Enum
from typing import List, Tuple, Optional

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class GameState(Enum):
    PLAYING = 1
    GAME_OVER = 2

class SnakeGame:
    def __init__(self, grid_size: int = 20, cell_size: int = 25):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Snake Game - Phase 1: Human Controls")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.DARK_GREEN = (0, 150, 0)
        
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state"""
        # Snake starts in the center, moving right
        center = self.grid_size // 2
        self.snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.score = 0
        self.game_state = GameState.PLAYING
        self.spawn_food()
    
    def spawn_food(self):
        """Spawn food at random location not occupied by snake"""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break
    
    def handle_input(self):
        """Handle keyboard input for snake direction"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.next_direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.next_direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.next_direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.next_direction = Direction.RIGHT
                elif event.key == pygame.K_SPACE and self.game_state == GameState.GAME_OVER:
                    self.reset_game()
        return True
    
    def update(self):
        """Update game state"""
        if self.game_state != GameState.PLAYING:
            return
        
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_state = GameState.GAME_OVER
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_state = GameState.GAME_OVER
            return
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
    
    def draw(self):
        """Draw the game"""
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            x, y = segment
            color = self.GREEN if i == 0 else self.DARK_GREEN  # Head is brighter
            pygame.draw.rect(self.screen, color, 
                           (x * self.cell_size, y * self.cell_size, 
                            self.cell_size, self.cell_size))
            
            # Add border to snake segments
            pygame.draw.rect(self.screen, self.BLACK,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size, self.cell_size), 1)
        
        # Draw food
        food_x, food_y = self.food
        pygame.draw.rect(self.screen, self.RED,
                        (food_x * self.cell_size, food_y * self.cell_size,
                         self.cell_size, self.cell_size))
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Draw game over message
        if self.game_state == GameState.GAME_OVER:
            game_over_text = self.font.render("GAME OVER", True, self.WHITE)
            restart_text = self.font.render("Press SPACE to restart", True, self.WHITE)
            
            # Center the text
            game_over_rect = game_over_text.get_rect(center=(self.window_size // 2, self.window_size // 2 - 20))
            restart_rect = restart_text.get_rect(center=(self.window_size // 2, self.window_size // 2 + 20))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(10)  # 10 FPS as specified
        
        pygame.quit()
        sys.exit()

    def get_game_info(self):
        """Get current game information for AI agents"""
        return {
            'snake': self.snake.copy(),
            'food': self.food,
            'score': self.score,
            'game_state': self.game_state,
            'grid_size': self.grid_size
        }

if __name__ == "__main__":
    game = SnakeGame()
    game.run()