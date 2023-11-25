import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

learn_rate=0.5
rewards=0.9
greedy=0.9
num_episode=200  # No. of episode

class QLearningTable:
    def __init__(self, actions, learning_rate=learn_rate, reward_decay=rewards, e_greedy=greedy):
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



        # Create obstacles (black rectangles)
        self.obstacle1 = self.canvas.create_rectangle(1 * UNIT, 1 * UNIT, 2 * UNIT, 2 * UNIT, fill='black')
        self.obstacle2 = self.canvas.create_rectangle(2 * UNIT, 1 * UNIT, 3 * UNIT, 2 * UNIT, fill='black')
        self.obstacle4 = self.canvas.create_rectangle(3 * UNIT, 1 * UNIT, 4 * UNIT, 2 * UNIT, fill='black')
        self.obstacle3 = self.canvas.create_rectangle(4 * UNIT, 1 * UNIT, 5 * UNIT, 2 * UNIT, fill='black')
        self.obstacle5 = self.canvas.create_rectangle(5 * UNIT, 1 * UNIT, 6 * UNIT, 2 * UNIT, fill='black')

        self.obstacle6 = self.canvas.create_rectangle(1 * UNIT, 2 * UNIT, 2 * UNIT, 3 * UNIT, fill='black')
        self.obstacle7 = self.canvas.create_rectangle(2 * UNIT, 2 * UNIT, 3 * UNIT, 3 * UNIT, fill='black')
        self.obstacle8 = self.canvas.create_rectangle(3 * UNIT, 2 * UNIT, 4 * UNIT, 3 * UNIT, fill='black')
        self.obstacle9 = self.canvas.create_rectangle(4 * UNIT, 2 * UNIT, 5 * UNIT, 3 * UNIT, fill='black')
        self.obstacle10 = self.canvas.create_rectangle(5 * UNIT, 2 * UNIT, 6 * UNIT, 3 * UNIT, fill='black')

        self.obstacle11 = self.canvas.create_rectangle(1 * UNIT, 3 * UNIT, 2 * UNIT, 4 * UNIT, fill='black')
        self.obstacle12 = self.canvas.create_rectangle(2 * UNIT, 3 * UNIT, 3 * UNIT, 4 * UNIT, fill='black')
        self.obstacle13 = self.canvas.create_rectangle(3 * UNIT, 3 * UNIT, 4 * UNIT, 4 * UNIT, fill='black')
        self.obstacle14 = self.canvas.create_rectangle(4 * UNIT, 3 * UNIT, 5 * UNIT, 4 * UNIT, fill='black')
        self.obstacle15 = self.canvas.create_rectangle(5 * UNIT, 3 * UNIT, 6 * UNIT, 4 * UNIT, fill='black')

        self.obstacle16 = self.canvas.create_rectangle(1 * UNIT, 4 * UNIT, 2 * UNIT, 5 * UNIT, fill='black')
        self.obstacle17 = self.canvas.create_rectangle(2 * UNIT, 4 * UNIT, 3 * UNIT, 5 * UNIT, fill='black')
        self.obstacle18 = self.canvas.create_rectangle(3 * UNIT, 4 * UNIT, 4 * UNIT, 5 * UNIT, fill='black')
        self.obstacle19 = self.canvas.create_rectangle(4 * UNIT, 4 * UNIT, 5 * UNIT, 5 * UNIT, fill='black')
        self.obstacle20 = self.canvas.create_rectangle(5 * UNIT, 4 * UNIT, 6 * UNIT, 5 * UNIT, fill='black')
        
        self.obstacle21 = self.canvas.create_rectangle(1 * UNIT, 5 * UNIT, 2 * UNIT, 6 * UNIT, fill='black')
        self.obstacle22 = self.canvas.create_rectangle(2 * UNIT, 5 * UNIT, 3 * UNIT, 6 * UNIT, fill='black')
        self.obstacle23 = self.canvas.create_rectangle(3 * UNIT, 5 * UNIT, 4 * UNIT, 6 * UNIT, fill='black')
        self.obstacle24 = self.canvas.create_rectangle(4 * UNIT, 5 * UNIT, 5 * UNIT, 6 * UNIT, fill='black')
        self.obstacle25 = self.canvas.create_rectangle(5 * UNIT, 5 * UNIT, 6 * UNIT, 6 * UNIT, fill='black')
        # self.obstacle23 = self.canvas.create_rectangle(9 * UNIT, 2 * UNIT, 10 * UNIT, 3 * UNIT, fill='black')

        

        # Create goal (yellow oval)
        self.goal = self.canvas.create_oval((WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT, WIDTH * UNIT, HEIGHT * UNIT, fill='orange')

        # Create agent (red rectangle)
        self.agent = self.canvas.create_rectangle(0, (HEIGHT - HEIGHT) * UNIT, (HEIGHT-6)*UNIT, (HEIGHT-6) * UNIT, fill='red')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(0, (HEIGHT - HEIGHT) * UNIT, (HEIGHT-6)*UNIT, (HEIGHT-6) * UNIT, fill='red')
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
        la=[self.canvas.coords(self.obstacle1), self.canvas.coords(self.obstacle2),self.canvas.coords(self.obstacle3),self.canvas.coords(self.obstacle4),self.canvas.coords(self.obstacle5),self.canvas.coords(self.obstacle6),self.canvas.coords(self.obstacle7),self.canvas.coords(self.obstacle8),self.canvas.coords(self.obstacle9), self.canvas.coords(self.obstacle10),self.canvas.coords(self.obstacle11),self.canvas.coords(self.obstacle12),self.canvas.coords(self.obstacle13), self.canvas.coords(self.obstacle14),self.canvas.coords(self.obstacle15),self.canvas.coords(self.obstacle16),self.canvas.coords(self.obstacle17), self.canvas.coords(self.obstacle18),self.canvas.coords(self.obstacle19),self.canvas.coords(self.obstacle20),self.canvas.coords(self.obstacle21),self.canvas.coords(self.obstacle22),self.canvas.coords(self.obstacle23),self.canvas.coords(self.obstacle24),self.canvas.coords(self.obstacle25)]
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

Episode_Times=[]
Total_Rewards = []
Pd_Errors = []
total_reward = 0
pd_error = 0
best_path = []
best_total_reward = 0
path=[]


def update():
    global total_reward 
    global pd_error
    global best_path
    global path

    best_total_reward = 0
    pos_pred= 0
    neg_pred= 0
    for episode in range(1,num_episode+1):
        start_time = time.time()
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            best_total_reward=reward
            total_reward += reward
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            path.append(observation)

            #print(path)
            with open('Q_table.txt','a') as f:
                f.write(str(RL.q_table) + '\n')


             # Calculate TD error
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


                print(f"Episode: {episode }, Total Reward: {total_reward}")
                # print("Q-table:")
                # print(RL.q_table)
                # print(f"pd_error: {pd_error}")

                # Check if the current path is the best
                if total_reward > best_total_reward:
                    best_total_reward = total_reward
                    best_path = path.copy()

                break

    print(f"Best Path: {best_path}")
    print(f"Number of Wrong Prediction: {neg_pred}")
    print(f"Number of Correct prediction: {pos_pred}")
    print(f"Learning Rate: {learn_rate}")
    print(f"Greedy Policy: {greedy}")
    print(f"Reward: {rewards}")
    print(f"No. of Episodes: {num_episode}")


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
    plt.title('Reward Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()


    plt.figure(figsize=(10, 5))
    plt.plot(Pd_Errors, label='PD Error', color='orange')
    plt.title('PD Error Vs Episode')
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
    #print('game over')
    env.destroy()

if __name__ == "__main__":
    WIDTH = 7
    HEIGHT = 7
    UNIT = 40

    env = RoomEnvironment()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

           
