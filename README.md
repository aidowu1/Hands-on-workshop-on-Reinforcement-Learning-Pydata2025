# Hands-on-workshop-on-Reinforcement-Learning-Pydata2025
Hands-on workshop @ Pydata 2025 London on developing Reinforcement Learning solutions with financial domain example use cases.

## Abstract
Reinforcement Learning (RL) has emerged as a transformative sub-field in AI/ML, driving breakthroughs in areas ranging from autonomous robotics to personalized recommendation systems. This workshop is designed to serve a broad audience—from beginners eager to grasp foundational RL concepts to practitioners seeking to deepen their technical expertise through applied projects. These projects will range from developing simple classical RL game environments to practical financial domain use cases such as using RL sequential decision making for stock trading and asset portfolio optimization scenarios.

Over the course of this interactive session, participants will embark on a journey that begins with an introduction to the fundamental principles of RL, including Markov Decision Processes, reward structures, and the critical balance between exploration and exploitation. We will then transition into a series of hands-on coding exercises using popular frameworks such as Python’s Gymnasium (formally referred to as Gym), PyTorch and RL open-source libraries such as Stable-baselines3 and Machin (to name a few). These exercises will enable attendees to implement classic algorithms like Q-learning, SARSA and deep learning algorithms such as actor-critic architectures and policy gradients in controlled environments.

Real-world case studies and example use cases—ranging from classical simple simulated game environments to realistic decision-making systems in finance (such as stock trading and asset portfolio optimization use cases) - will illustrate how RL methodologies are applied in practice. During this workshop participants will develop and fine-tune RL models, gaining insights into performance evaluation, model tuning, and deployment strategies. Additionally, advanced topics such as deep RL architectures, on-policy and off-policy RL algorithms will be discussed and hacked interactively.

This workshop aims not only to impart theoretical knowledge but also to empower participants with the practical skills needed to design and deploy effective RL solutions. Join us to explore the dynamic world of reinforcement learning and to enhance your toolkit for solving complex, data-driven challenges. All the python libraries/packages, reference papers and data used in this workshop will be open sourced and made available in a Github repo (which will be made available soon).

### Demos
There will be 5 demos:
 - The first 2 is an introduction to RL and uses Q-Learning, SARSA and DQN (these were coded natively in python with no RL packages)
 - The third one is an introduction to continuous action space problems and uses 3rd party open source libraries Machin and Stable Baselines
 -The fourth demo introduces us to hyper-parameter tuning using Stable Baseline library
 - The final demo is the main one which demonstrates a simple algo RL agent for trading the S & P index using price/bar data from  2010 to 2019

### Instructions on setting-up
Install the code from the git repo:
 - Launch your local version of command prompt (cmd)
 - Clone the code repo by typing the following in the CMD prompt:
   - git clone git@github.com:aidowu1/Hands-on-workshop-on-Reinforcement-Learning-Pydata2025.git
 - Change your working directory to the root folder of the application:
   - cd Hands-on-workshop-on-Reinforcement-Learning-Pydata2025
 - Ensure you have a python distribution on your PC, for this solution I used the Miniconda distribution. Please following the instruction described here to install minconda on your PC.
 - Create a python virtual environment for this solution following these steps (I used a miniconda virtual environment with python 3.11 and a requirements.txt file pulled from the repo):
   - conda create -n hedging_env python=3.11 -y
   - Activate the newly created environment:
     - conda activate rl_env
   - Install other python packages in the activated environment using requirements.txt obtained from the repo:
     - pip install -r ./demos/requirements.txt 
 

Demo source code:
 - Source code for the Grid World demo is located here:
   - ./demos/grid-world
 - Source code for the Frozen Lake demo is located here: 
   - ./demos/frozen-lake
 - Source code for the Cart Pole demo is located here:
   - [cart-pole](https://github.com/aidowu1/Hands-on-workshop-on-Reinforcement-Learning-Pydata2025/tree/main/demos/cart-pole)
 - Source code for the Pendulum demo is located here:
   - ./demos/pendulum
 - Source code for the Algorithmic RL trader demo is located here:
   - ./demos/rl-robo-algo-trader

