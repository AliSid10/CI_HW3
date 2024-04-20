import numpy as np
import gym
import random
import math
env1 =["SHFF", "FFFH", "FHFH", "HFFG"]
env2= ["SFFFFF", "FFFHFF", "FHFHHH", "HFFFFG"]
env3 = ['SFFHFFHH', 'HFFFFFHF', 'HFFHHFHH', 'HFHHHFFF', 'HFHHFHFF', 'FFFFFFFH', 'FHHFHFHH', 'FHHFHFFG'] 

selectedEnv = env2
env = gym.make('FrozenLake-v1', desc=selectedEnv, render_mode="human", is_slippery = False)
env.reset()
env.render()
# change-able parameters:
discount_factor = 0.99
delta_threshold = 0.00001
epsilon = 1

def value_iteration(env, gamma=0.9, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(num_states)
    # Initialize the value function
    V = np.zeros(num_states)
    delta1=float("inf")
    while delta1>=epsilon:
       print(delta1)
       delta1 = 0
       for s in range(num_states):
           v = V[s]
           max_v = -float("inf")
           for a in range(num_actions):
                temp = 0
                for tup in env.P[s][a]:
                    p = tup[0]
                    next_state = tup[1]
                    reward = tup[2]
                    temp += p*(reward+gamma*V[next_state])
                   
                if temp > max_v:
                    max_v = temp
           V[s] = max_v
           delta1 = max(abs(v-V[s]),delta1)

    Pol = np.zeros(num_states, dtype=int)
    for s in range(num_states):
           v = V[s]
           max_v = -float("inf")
           max_a = None
           for a in range(num_actions):
                temp = 0
                for tup in env.P[s][a]:
                    p = tup[0]
                    next_state = tup[1]
                    reward = tup[2]
                    temp += p*(reward+gamma*V[next_state])
                   
                if temp > max_v:
                    max_v = temp
                    max_a = a
           Pol[s] = max_a

    return Pol, V



                

                   
policy, V = value_iteration(env)


# Print results
print("Optimal Value Function:")
print(V.reshape(len(selectedEnv), len(selectedEnv[0])))

print("\nOptimal Policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(len(selectedEnv), len(selectedEnv[0])))

# resetting the environment and executing the policy
state = env.reset()
state = state[0]
step = 0
done = False
print(state)

max_steps = 100
for step in range(max_steps):

    # Getting max value against that state, so that we choose that action
    action =policy[state]
    new_state, reward, done, truncated, info = env.step(action) #information after taking the action

    env.render()
    if done:
        print("number of steps taken:", step)
        break

    state = new_state

env.close()

