import numpy as np 
import pandas as pd 
import time 

np.random.seed(2)

N_STATES = 6 # the length of the space
ACTIONS = ['left','right'] # availiable actions
EPSILON = 0.9 # greedy police
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # the discount factor for the R update
MAX_EPISODES = 13 # MAX episods
FRESH_TIME = 0.1 # fresh time after each move

def build_q_table(n_states,actions):
    '''
    init the q table
    '''
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions)
    return table

def choose_action(state,q_table):
    '''
    choose the next action
    '''
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform()>EPSILON) or ((state_actions == 0).all(0)): # random choose action
        action_name = np.random.choice(ACTIONS)
    else: # choose the max q_value action
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S,A):
    '''
    update the state
    '''
    if A == 'right':
        if S == N_STATES -2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:# move right
        R = 0
        if S == 0: # reach the wall
            S_ = S 
        else:
            S_ = S - 1
    return S_, R

def update_env(S,episode,step_counter):
    '''
    update the env
    '''
    env_list = ['-'] * (N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode  %s: total steps = %s'%(episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                          ',end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def ql():
    '''
    the q_learning function
    '''
    q_table = build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminal = False
        update_env(S,episode,step_counter)

        while not is_terminal:
            A = choose_action(S,q_table)
            S_,R = get_env_feedback(S,A)
            q_predict = q_table.loc[S,A]
            #update the q value
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminal = True
            
            q_table.loc[S,A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S,episode,step_counter+1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = ql()
    print('\r\nQ-table:\n')
    print(q_table)

