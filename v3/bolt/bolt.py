from agent0 import Agent  # Replace with your agent file name and class
from env import CustomEnv

def main():
    env = CustomEnv()  # Initialize the environment
    agent = Agent(env)  # Initialize the agent with the environment
    agent.train()  # Begin training the agent
    agent.test()  # Test the agent

if __name__ == "__main__":
    main()