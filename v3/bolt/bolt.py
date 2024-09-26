# Replace with your agent file name and class
from agent0 import Agent  
from env import CustomEnv

def main():
    # Initialize the environment
    env = CustomEnv()  
    # Initialize the agent with the environment
    agent = Agent(env) 
    # Begin training the agent
    agent.train()  
    # Test the agent
    agent.test() 

if __name__ == "__main__":
    main()
