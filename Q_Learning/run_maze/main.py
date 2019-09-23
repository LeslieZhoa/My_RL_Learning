from maze_env import Maze 
from RL_brain import QLearningTable 

def update():
    for episode in range(100):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))

            observation_,reward,done = env.step(action)

            RL.learn(str(observation),action,reward,str(observation_))
            # print(str(observation))

            observation = observation_

            if done:
                break 

    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()