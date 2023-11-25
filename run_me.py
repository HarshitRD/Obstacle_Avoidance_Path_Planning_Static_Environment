import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

#initializing the Parameters
learn_rate=0.1
reward_d=0.9
greedy=0.9

#Q learning Table
class QLTable:
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

# create the room environment 
class RoomEnvironment(tk.Tk, object):
    def __init__(self):
        super(RoomEnvironment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('room_environment')
        self.geometry('{0}x{1}'.format(Width * Unit, Height * Unit))
        self._build_environment()

    def _build_environment(self):

        #set the Canvas
        self.canvas = tk.Canvas(self, bg='white', height=Height * Unit, width=Width * Unit)
        MAZE_W=Width
        MAZE_H=Height

        #to see the grid
        # for c in range(0, MAZE_W * Unit, Unit):
        #     x0, y0, x1, y1 = c, 0, c, MAZE_H * Unit
        #     self.canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, MAZE_H * Unit, Unit):
        #     x0, y0, x1, y1 = 0, r, MAZE_W * Unit, r
        #     self.canvas.create_line(x0, y0, x1, y1)

        # Create obstacles (black rectangles)

        self.ob2 = self.canvas.create_rectangle(3 * Unit, 0 * Unit, 4 * Unit, 1 * Unit, fill='black')
        self.ob4 = self.canvas.create_rectangle(3 * Unit, 1 * Unit, 4 * Unit, 2 * Unit, fill='black')
        self.ob3 = self.canvas.create_rectangle(3 * Unit, 2 * Unit, 4 * Unit, 3 * Unit, fill='black')
        
        self.ob16 = self.canvas.create_rectangle(6 * Unit, 1 * Unit, 7 * Unit, 2 * Unit, fill='black')
        self.ob15 = self.canvas.create_rectangle(6 * Unit, 3 * Unit, 7 * Unit, 4 * Unit, fill='black')
        self.ob1 = self.canvas.create_rectangle(2 * Unit, 7 * Unit, 3 * Unit, 8 * Unit, fill='black')
        self.ob5 = self.canvas.create_rectangle(3 * Unit, 6 * Unit, 4 * Unit, 7 * Unit, fill='black')
        self.ob6 = self.canvas.create_rectangle(2 * Unit, 6 * Unit, 3 * Unit, 7 * Unit, fill='black')
        self.ob7 = self.canvas.create_rectangle(3 * Unit, 7 * Unit, 4 * Unit, 8 * Unit, fill='black')
        self.ob8 = self.canvas.create_rectangle(6 * Unit, 7 * Unit, 7 * Unit, 8 * Unit, fill='black')
        self.ob9 = self.canvas.create_rectangle(6 * Unit, 8 * Unit, 7 * Unit, 9 * Unit, fill='black')
        self.ob10 = self.canvas.create_rectangle(6 * Unit, 9 * Unit, 7 * Unit, 10 * Unit, fill='black')
        
        self.ob11 = self.canvas.create_rectangle(9 * Unit, 2 * Unit, 10 * Unit, 3 * Unit, fill='black')
        self.ob12 = self.canvas.create_rectangle(9 * Unit, 2 * Unit, 10 * Unit, 3 * Unit, fill='black')

        self.ob13 = self.canvas.create_rectangle(6 * Unit, 9 * Unit, 7 * Unit, 10 * Unit, fill='black')
        self.ob14 = self.canvas.create_rectangle( 0* Unit, 9 * Unit, 1 * Unit, 10 * Unit, fill='black')

        # Create Goal location
        self.goal = self.canvas.create_oval((Width - 1) * Unit, (Height - 1) * Unit, Width * Unit, Height * Unit, fill='orange')

        # Create agent 
        self.agent = self.canvas.create_rectangle(0, (Height - Height) * Unit, (Height-9)*Unit, (Height-9) * Unit, fill='red')

        self.canvas.pack()
    
    # To reset the agent to its initial position
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(0, (Height - Height) * Unit, (Height-9)*Unit, (Height-9) * Unit, fill='red')
        return self.canvas.coords(self.agent)
    
    
    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        if action == 0:  # Up
            if s[1] > Unit:
                base_action[1] -= Unit
        elif action == 1:  # Down
            if s[1] < (Height - 1) * Unit:
                base_action[1] += Unit
        elif action == 2:  # Left
            if s[0] > 0:
                base_action[0] -= Unit
        elif action == 3:  # Right
            if s[0] < (Width - 1) * Unit:
                base_action[0] += Unit

        self.canvas.move(self.agent, base_action[0], base_action[1])
        
        la=[self.canvas.coords(self.ob1),self.canvas.coords(self.ob2),self.canvas.coords(self.ob3),self.canvas.coords(self.ob4),self.canvas.coords(self.ob5),self.canvas.coords(self.ob6),self.canvas.coords(self.ob7),self.canvas.coords(self.ob8),self.canvas.coords(self.ob9),self.canvas.coords(self.ob10),self.canvas.coords(self.ob11),self.canvas.coords(self.ob12),self.canvas.coords(self.ob13), self.canvas.coords(self.ob14),self.canvas.coords(self.ob15),self.canvas.coords(self.ob16)]
       
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

T_Rewards = [] #to store the all reward points
Pd_E = [] #Prediction steps are correct are wrong
Epi_Times = [] #timw spent by episode
total_reward = 0
pd_e = 0
b_path = [] #path which give positive reward
b_total_reward = 0
path=[] #store all the path
Num_Episode=100 #number of episodes


# Update the movement of the agent
def update():
    global total_reward 
    global pd_e
    global b_path
    global path
    b_total_reward = 0
    pos_pred=0
    neg_pred=0
    for episode in range(1,Num_Episode+1):
        s_time = time.time() #start the time
        observation = env.reset() 

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
             
            total_reward=total_reward + reward 
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            path.append(observation) 
            #print(path)
            with open('Q_table.txt', 'a') as f:   #store the Q_table to txt file
                f.write(str(RL.q_table) + '\n')

            # Calculate Predection error
            q_pred = RL.q_table.loc[str(observation), action]
            if done:
                q_t = reward
            else:
                q_t = reward + RL.gamma * RL.q_table.loc[str(observation_), :].max()

            pd_e = q_t - q_pred 

            if done:
                e_time = time.time()  #end the time when episode is completed
                epi_time = e_time - s_time  #time required for the episode
                Epi_Times.append(epi_time) 
                T_Rewards.append(total_reward)
                Pd_E.append(pd_e)
                if pd_e<0:
                    neg_pred=neg_pred+1 #number of wrong predection or steps

                if pd_e>0:
                    pos_pred=pos_pred+1 #number of correct predection or steps

                print(f"Episode: {episode}, Total Reward: {total_reward}")
                #print("Q-table:")
                #print(RL.q_table)
                #print(f"pd_error: {pd_error}")
                if reward > 0: 
                    #b_total_reward = total_reward
                    b_path=path.copy()   #store the path for every positive reward
                    #print(f"best_path:{best_path}")
                break
    
    print(f"learning_rate={learn_rate}, reward_decay={reward_d}, e_greedy={greedy},episode={Num_Episode}")
    print(f"Best Path: {b_path}")
    print(f"Number of wrong Prediction: {neg_pred}")
    print(f"Number of Correct Prediction: {pos_pred}")
    
    #store the values in text file
    with open('total_rewards.txt', 'a') as f:
        f.write(str(T_Rewards) + '\n')

    with open('pd_errors.txt', 'a') as f:
        f.write(str(Pd_E) + '\n')

    with open('episode_times.txt', 'a') as f:
        f.write(str(Epi_Times) + '\n')
    
    with open('Best_path.txt', 'a') as f:
        f.write(str(b_path) + '\n')
        
    #plot the Reward, Time per episode and Prediction
        
    plt.figure(figsize=(10, 5))
    plt.plot(T_Rewards, label='Total Reward')
    plt.title('Rewards Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(Pd_E, label='PD Error', color='orange')
    plt.title('Prediction Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Prediction')
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(Epi_Times, label='Episode Time', color='green')
    plt.title('Time Vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print('game over')
    env.destroy()

if __name__ == "__main__":
    Width = 10  #Width of the environment
    Height = 10  #height of the complete environment
    Unit = 40    #Size of the one block

    env = RoomEnvironment()
    RL = QLTable(actions=list(range(env.n_actions)))

    env.after(100, update) #update after 100ms
    env.mainloop()

           
