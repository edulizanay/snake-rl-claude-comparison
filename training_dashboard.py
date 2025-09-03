import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle
import os
from qlearning_agent import QLearningAgent
from random_agent import RandomAgent, GameRunner

class TrainingDashboard:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Snake Q-Learning Training Dashboard', fontsize=16)
        
    def plot_training_progress(self, agent, window_size=100):
        """Plot training progress with rolling average"""
        ax = self.axes[0, 0]
        ax.clear()
        
        episodes = agent.training_episodes
        scores = agent.training_scores
        
        if len(scores) > 0:
            # Plot raw scores
            ax.plot(episodes, scores, alpha=0.3, color='blue', label='Raw scores')
            
            # Plot rolling average
            if len(scores) >= window_size:
                rolling_avg = []
                for i in range(window_size-1, len(scores)):
                    avg = np.mean(scores[i-window_size+1:i+1])
                    rolling_avg.append(avg)
                
                rolling_episodes = episodes[window_size-1:]
                ax.plot(rolling_episodes, rolling_avg, color='red', linewidth=2, 
                       label=f'Rolling Average ({window_size} episodes)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)
    
    def plot_epsilon_decay(self, agent):
        """Plot epsilon decay over time"""
        ax = self.axes[0, 1]
        ax.clear()
        
        episodes = agent.training_episodes
        epsilon_history = agent.epsilon_history
        
        if len(epsilon_history) > 0:
            ax.plot(episodes, epsilon_history, color='green', linewidth=2)
            ax.axhline(y=agent.epsilon_min, color='red', linestyle='--', 
                      label=f'Min Epsilon ({agent.epsilon_min})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon) Decay')
        ax.legend()
        ax.grid(True)
    
    def plot_q_table_heatmap(self, agent, top_states=20):
        """Plot heatmap of Q-table values for most visited states"""
        ax = self.axes[1, 0]
        ax.clear()
        
        # Get most common states
        state_counts = defaultdict(int)
        for state in agent.q_table.keys():
            state_counts[state] += len(agent.q_table[state])
        
        if len(state_counts) == 0:
            ax.text(0.5, 0.5, 'No Q-table data yet', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Q-Table Heatmap (Top States)')
            return
        
        # Sort by frequency and take top states
        top_states_list = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:top_states]
        
        # Create heatmap data
        state_names = []
        q_values_matrix = []
        
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for state, _ in top_states_list:
            # Create readable state name
            food_dir, danger, wall_dist, curr_dir = state
            state_name = f"{food_dir[:2]}-{danger}-{wall_dist}-{curr_dir[:1]}"
            state_names.append(state_name)
            
            # Get Q-values for this state
            q_vals = []
            for action_name in actions:
                # Find matching direction
                from snake_game import Direction
                action_mapping = {
                    'UP': Direction.UP,
                    'DOWN': Direction.DOWN, 
                    'LEFT': Direction.LEFT,
                    'RIGHT': Direction.RIGHT
                }
                action = action_mapping[action_name]
                q_vals.append(agent.q_table[state][action])
            
            q_values_matrix.append(q_vals)
        
        if len(q_values_matrix) > 0:
            # Create heatmap
            im = ax.imshow(q_values_matrix, cmap='RdYlBu', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(actions)))
            ax.set_xticklabels(actions)
            ax.set_yticks(range(len(state_names)))
            ax.set_yticklabels(state_names, fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.6)
            
            # Add text annotations
            for i in range(len(state_names)):
                for j in range(len(actions)):
                    if len(q_values_matrix) > i and len(q_values_matrix[i]) > j:
                        text = ax.text(j, i, f'{q_values_matrix[i][j]:.2f}', 
                                     ha='center', va='center', fontsize=6)
        
        ax.set_title('Q-Table Heatmap (Top States)')
    
    def plot_performance_comparison(self, qlearning_stats, random_stats):
        """Plot comparison between Q-learning and random baseline"""
        ax = self.axes[1, 1]
        ax.clear()
        
        categories = ['Mean Score', 'Max Score', 'Mean Steps']
        ql_values = [qlearning_stats['mean_score'], qlearning_stats['max_score'], 
                    qlearning_stats['mean_steps']]
        random_values = [random_stats['mean_score'], random_stats['max_score'], 
                        random_stats['mean_steps']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ql_values, width, label='Q-Learning', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, random_values, width, label='Random', color='red', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        
        ax.grid(True, alpha=0.3)
    
    def update_dashboard(self, agent, qlearning_stats=None, random_stats=None):
        """Update all plots in the dashboard"""
        self.plot_training_progress(agent)
        self.plot_epsilon_decay(agent)
        self.plot_q_table_heatmap(agent)
        
        if qlearning_stats and random_stats:
            self.plot_performance_comparison(qlearning_stats, random_stats)
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def save_dashboard(self, filename='training_dashboard.png'):
        """Save the dashboard as an image"""
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved as {filename}")
    
    def show(self):
        """Show the dashboard"""
        plt.show()

def train_with_dashboard():
    """Train Q-learning agent with live dashboard updates"""
    print("=== Q-Learning Training with Live Dashboard ===")
    
    # Initialize agent and dashboard
    agent = QLearningAgent()
    dashboard = TrainingDashboard()
    
    # Get random baseline for comparison
    print("Establishing random baseline...")
    random_agent = RandomAgent()
    runner = GameRunner(show_gui=False)
    random_stats, _, _ = runner.run_multiple_games(random_agent, num_games=100)
    print(f"Random baseline: {random_stats['mean_score']:.2f} average score")
    
    # Training parameters
    total_episodes = 1000
    update_interval = 50
    
    print(f"Training for {total_episodes} episodes with dashboard updates every {update_interval} episodes...")
    
    plt.ion()  # Interactive mode on
    
    for episode in range(0, total_episodes, update_interval):
        # Train for a batch of episodes
        batch_episodes = min(update_interval, total_episodes - episode)
        print(f"Training episodes {episode} to {episode + batch_episodes}...")
        
        for i in range(batch_episodes):
            score, steps = agent.train_episode()
            agent.training_scores.append(score)
            agent.training_episodes.append(episode + i)
            agent.epsilon_history.append(agent.epsilon)
            
            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
        
        # Update dashboard
        dashboard.update_dashboard(agent)
        
        # Show progress
        recent_scores = agent.training_scores[-batch_episodes:]
        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        print(f"Episodes {episode}-{episode + batch_episodes}: Avg Score = {avg_score:.2f}, Epsilon = {agent.epsilon:.3f}")
    
    # Final evaluation
    print("Training completed! Evaluating final performance...")
    qlearning_stats, scores, steps = agent.evaluate(num_games=100)
    
    # Final dashboard update
    dashboard.update_dashboard(agent, qlearning_stats, random_stats)
    
    # Show results
    print("\n=== Final Results ===")
    print(f"Q-Learning Average Score: {qlearning_stats['mean_score']:.2f}")
    print(f"Random Baseline Average Score: {random_stats['mean_score']:.2f}")
    print(f"Improvement: {((qlearning_stats['mean_score'] - random_stats['mean_score']) / random_stats['mean_score'] * 100):.1f}%")
    
    # Save model and dashboard
    agent.save("snake_qlearning_trained.pkl")
    dashboard.save_dashboard()
    
    # Keep dashboard open
    plt.ioff()
    dashboard.show()
    
    return agent, qlearning_stats, random_stats

def load_and_visualize():
    """Load existing model and show dashboard"""
    filename = input("Enter model filename: ").strip()
    if not os.path.exists(filename):
        print("File not found!")
        return
    
    agent = QLearningAgent()
    agent.load(filename)
    
    # Get stats
    qlearning_stats, _, _ = agent.evaluate(num_games=100)
    
    # Get random baseline
    random_agent = RandomAgent()
    runner = GameRunner(show_gui=False)
    random_stats, _, _ = runner.run_multiple_games(random_agent, num_games=100)
    
    # Show dashboard
    dashboard = TrainingDashboard()
    dashboard.update_dashboard(agent, qlearning_stats, random_stats)
    dashboard.show()

def main():
    print("=== Training Dashboard ===")
    print("1. Train new agent with live dashboard")
    print("2. Load existing model and show dashboard")
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "1":
        train_with_dashboard()
    elif choice == "2":
        load_and_visualize()
    else:
        print("Invalid choice. Starting training with dashboard...")
        train_with_dashboard()

if __name__ == "__main__":
    main()