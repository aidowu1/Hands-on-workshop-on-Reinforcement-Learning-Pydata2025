from src.main.rl_agents.random_agent import RandomAgent
from src.main.rl_agents.q_learning_agent import QLearningAgent

def executeRandomGridWorldAgent():
    """
    Execute the random Grid World agent
    :return:
    """
    agent = RandomAgent()
    avg_reward_per_episode, cum_reward_per_episode, steps_per_episode = agent.run()
    agent.plotRewardCurves()

def executeQLearningGridWorldAgent():
    """
    Execute the Q-Learning Grid World agent
    :return:
    """
    agent = QLearningAgent(render_mode="rgb_array")
    avg_reward_per_episode, cum_reward_per_episode, steps_per_episode = agent.run()
    agent.plotRewardCurves()

if __name__ == "__main__":
    # executeRandomGridWorldAgent()
    executeQLearningGridWorldAgent()
