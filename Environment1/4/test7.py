import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

learn_rate=0.5
reward_d=0.5
greedy=0.5
class QLearningTable:
    global learn_rate
    global greedy
    global reward_d
    def __init__(self, actions, learning_rate=learn_rate, reward_decay=reward_d, e_greedy=greedy):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            if state == 'goal':
                self.q_table = self.q_table._append(pd.Series([10] * len(self.actions), index=self.q_table.columns, name=state))
            else:
                self.q_table = self.q_table._append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class RoomEnvironment(tk.Tk, object):
    def __init__(self):
        super(RoomEnvironment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('room_environment')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT))
        self._build_environment()

    def _build_environment(self):
        self.canvas = tk.Canvas(self, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        MAZE_W=WIDTH
        MAZE_H=HEIGHT

        # for c in range(0, MAZE_W * UNIT, UNIT):
        #     x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
        #     self.canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, MAZE_H * UNIT, UNIT):
        #     x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
        #     self.canvas.create_line(x0, y0, x1, y1)

        # Create obstacles (black rectangles)
        #self.obstacle1 = self.canvas.create_rectangle(0 * UNIT, 2 * UNIT, 1 * UNIT, 3 * UNIT, fill='black')

        # Create obstacles (black rectangles)
        self.obstacle1 = self.canvas.create_rectangle(3 * UNIT, 3 * UNIT, 4 * UNIT, 4 * UNIT,fill='black')
        self.obstacle2 = self.canvas.create_rectangle(1 * UNIT, 2 * UNIT, 2 * UNIT, 3 * UNIT, fill='black')
        self.obstacle3 = self.canvas.create_rectangle(2 * UNIT, 1 * UNIT, 3 * UNIT, 2 * UNIT, fill='black')
        self.goal = self.canvas.create_oval((WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT, WIDTH * UNIT, HEIGHT * UNIT, fill='orange')

        # Create agent (red rectangle)
        self.agent = self.canvas.create_rectangle(0, (HEIGHT - HEIGHT) * UNIT, (HEIGHT-4)*UNIT, (HEIGHT-4) * UNIT, fill='red')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(0, (HEIGHT - HEIGHT) * UNIT, (HEIGHT-4)*UNIT, (HEIGHT-4) * UNIT, fill='red')
        return self.canvas.coords(self.agent)

    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # Up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # Down
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # Left
            if s[0] > 0:
                base_action[0] -= UNIT
        elif action == 3:  # Right
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])
        la=[self.canvas.coords(self.obstacle1),self.canvas.coords(self.obstacle2),self.canvas.coords(self.obstacle3)]
        s_ = self.canvas.coords(self.agent)
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in la:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


Total_Rewards = []
Pd_Errors = []
Episode_Times = []
total_reward = 0
pd_error = 0
best_path = []
best_total_reward = 0
path=[]
Num_Episode=100

def update():
    global total_reward 
    global pd_error
    global best_path
    global path
    best_total_reward = 0
    pos_pred=0
    neg_pred=0
    for episode in range(1,Num_Episode+1):
        start_time = time.time()
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            
            total_reward += reward
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            path.append(observation)
            #print(path)
            with open('Q_table.txt', 'a') as f:
                f.write(str(RL.q_table) + '\n')

             # Calculate PD error
            q_predict = RL.q_table.loc[str(observation), action]
            if done:
                q_target = reward
            else:
                q_target = reward + RL.gamma * RL.q_table.loc[str(observation_), :].max()

            pd_error = q_target - q_predict


            if done:
                end_time = time.time()
                episode_time = end_time - start_time
                Episode_Times.append(episode_time)
                Total_Rewards.append(total_reward)
                Pd_Errors.append(pd_error)
                if pd_error<0:
                    neg_pred=neg_pred+1

                if pd_error>0:
                    pos_pred=pos_pred+1

                print(f"Episode: {episode}, Total Reward: {total_reward}")
                #print("Q-table:")
                #print(RL.q_table)
                #print(f"pd_error: {pd_error}")

                # Check if the current path is the best
                if total_reward > best_total_reward:
                    best_total_reward = total_reward
                    best_path=path.copy()
                    #print(f"best_path:{best_path}")
                break
    
    print(f"learning_rate={learn_rate}, reward_decay={reward_d}, e_greedy={greedy},episode={Num_Episode}")
    print(f"Best Path: {best_path}")
    print(f"Number of wrong Prediction: {neg_pred}")
    print(f"Number of Correct Prediction: {pos_pred}")
    

    with open('total_rewards.txt', 'a') as f:
        f.write(str(Total_Rewards) + '\n')

    with open('pd_errors.txt', 'a') as f:
        f.write(str(Pd_Errors) + '\n')

    with open('episode_times.txt', 'a') as f:
        f.write(str(Episode_Times) + '\n')
    
    with open('Best_path.txt', 'a') as f:
        f.write(str(best_path) + '\n')
        
    plt.figure(figsize=(10, 5))
    plt.plot(Total_Rewards, label='Total Reward')
    plt.title('Rewards Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(Pd_Errors, label='PD Error', color='orange')
    plt.title('Prediction Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('PD Error')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(Episode_Times, label='Episode Time', color='green')
    plt.title('Time Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print('game over')
    env.destroy()

if __name__ == "__main__":
    WIDTH = 5
    HEIGHT = 5
    UNIT = 40

    env = RoomEnvironment()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

           
