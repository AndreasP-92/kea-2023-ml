import numpy as np
import random

# initialize Q-table
Q = np.zeros([16, 4])

# set hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 10000

# define function to choose an action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state, :])

# define function to update Q-table
def update_Q_table(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

# loop over episodes
for episode in range(num_episodes):
    # reset environment
    state = 0
    done = False
    
    # loop over time steps within episode
    while not done:
        # choose action
        action = choose_action(state)
        
        # take action and observe reward and next state
        if action == 0:
            if state % 4 == 0:
                reward = -1
                next_state = state
            else:
                next_state = state - 1
                if next_state in [5, 7, 11, 12]:
                    reward = -1
                elif next_state == 15:
                    reward = 1
                else:
                    reward = 0
        elif action == 1:
            if state > 11:
                reward = -1
                next_state = state
            else:
                next_state = state + 4
                if next_state in [5, 7, 11, 12]:
                    reward = -1
                elif next_state == 15:
                    reward = 1
                else:
                    reward = 0
        elif action == 2:
            if state % 4 == 3:
                reward = -1
                next_state = state
            else:
                next_state = state + 1
                if next_state in [5, 7, 11, 12]:
                    reward = -1
                elif next_state == 15:
                    reward = 1
                else:
                    reward = 0
        elif action == 3:
            if state < 4:
                reward = -1
                next_state = state
            else:
                next_state = state - 4
                if next_state in [5, 7, 11, 12]:
                    reward = -1
                elif next_state == 15:
                    reward = 1
                else:
                    reward = 0
        
        # update Q-table
        update_Q_table(state, action, reward, next_state)
        
        # update state and check if episode is done
        state = next_state
        if state in [5, 7, 11, 12, 15]:
            done = True
            
# play game using learned Q-table
state = 0
done = False
while not done:
    action = np.argmax(Q[state, :])
    if action == 0:
        if state % 4 == 0:
            next_state = state
        else:
            next_state = state - 1
    elif action == 1:
        if state > 11:
            next_state = state
        else:
            next_state = state + 4
    elif action == 2:
        if state % 4 == 3:
            next_state = state
        else:
            next_state = state + 1
    elif action == 3:
        if state < 4:
            next_state = state
        else:
            next_state = state - 4
    state = next_state
    if state == 15:
        done = True
print("Game over, you won!")
