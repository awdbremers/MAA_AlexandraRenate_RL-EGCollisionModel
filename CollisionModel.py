%matplotlib inline

import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import numpy as np
import random

def compute_avgreward(model):
    avgreward = [agent.reward for agent in model.schedule.agents]
    x = sorted(avgreward)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)+0.0001)
    return sum(avgreward) / len(avgreward)

def compute_avg_reward_angle_0(model):
    avgreward = [agent.A_r[0] for agent in model.schedule.agents]
    x = sorted(avgreward)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)+0.0001)
    return sum(avgreward) / len(avgreward)

def compute_avg_reward_angle_1(model):
    avgreward = [agent.A_r[1] for agent in model.schedule.agents]
    x = sorted(avgreward)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)+0.0001)
    return sum(avgreward) / len(avgreward)

def compute_avg_reward_angle_2(model):
    avgreward = [agent.A_r[2] for agent in model.schedule.agents]
    x = sorted(avgreward)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)+0.0001)
    return sum(avgreward) / len(avgreward)

def compute_avg_reward_angle_3(model):
    avgreward = [agent.A_r[3] for agent in model.schedule.agents]
    x = sorted(avgreward)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x)+0.0001)
    return sum(avgreward) / len(avgreward)

def compute_avgcollision(model):
    avgcollision = [agent.collision for agent in model.schedule.agents]
    x = sorted(avgcollision)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return sum(avgcollision) / len(avgcollision)

class CollisionModel(Model):
    def __init__(self, N, width, height, init_value):
        self.num_agents = N
        self.init_value = init_value
        self.grid = MultiGrid(width, height, True)
        self.schedule = BaseScheduler(self)
        
        # Create Agents
        for i in range(self.num_agents):
            a = CollisionAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        
        self.datacollector = DataCollector(
            #model_reporters={"AvgReward": compute_avgreward},
            #model_reporters={"AvgCollision": compute_avgcollision},
            model_reporters={"0": compute_avg_reward_angle_0, "90": compute_avg_reward_angle_1, "180": compute_avg_reward_angle_2, "270": compute_avg_reward_angle_3},
            agent_reporters={"Reward": lambda a: a.reward})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

class CollisionAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.reward = init_value
        self.collision = 1 #this initial value should be zero, but is set to 1 to avoid zero division (-1 later on)
        self.A_r = [0,0,0,0] #this should be updated after a collision/succesfull move 0,90,180,270 degrees
        self.A_p = [0,0,0,0] #this stores how many times an angle was chosen
        self.prevdir = 0 #this is replaced 
        self.A_0 = [0,2,3,1] #calculate direction from previous direction (list) and angle (index)
        self.A_1 = [1,0,2,3]
        self.A_2 = [2,3,1,0]
        self.A_3 = [3,1,0,2]
        
    def move(self):
        epsilon = 0.30
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=False)
        
        ### e-greedy is introduced as an if statement. here, first the angle is either chosen or generated
                 
        if np.random.random() > epsilon and self.A_r != init_dir_r:
            # Exploit (use best angle)
            self.angle = np.argmax(self.A_r)
            angle = self.angle
            angledegrees = (self.angle)*90
            # calculate direction from angle
            if self.prevdir == 0:
                self.dir = self.A_0[angle]
            if self.prevdir == 1:
                self.dir = self.A_1[angle]
            if self.prevdir == 2:
                self.dir = self.A_2[angle]
            if self.prevdir == 3:
                self.dir = self.A_3[angle]
            dir = self.dir
            
            new_position = possible_steps[dir] #picks actual coordinates corresponding with direction
            occupation = self.model.grid.get_cell_list_contents([new_position])
            if len(occupation) < 1:
                self.model.grid.move_agent(self, new_position)
                self.A_r[angle] += R1
                self.A_p[angle] += 1
                #print("Agent",self.unique_id,angledegrees,"Exploited-Success")
            else:
                self.A_r[angle] += R2
                self.A_p[angle] += 1
                self.collision += 1
                #print("Agent",self.unique_id,angledegrees,"Exploited-COLLIDE")
        else:
            # Explore (test all directions)
            self.angle = random.choice(angles)
            angle = self.angle
            angledegrees = (self.angle)*90
            # calculate direction from angle
            if self.prevdir == 0:
                self.dir = self.A_0[angle]
            if self.prevdir == 1:
                self.dir = self.A_1[angle]
            if self.prevdir == 2:
                self.dir = self.A_2[angle]
            if self.prevdir == 3:
                self.dir = self.A_3[angle]
            dir = self.dir
            
            
            new_position = possible_steps[dir]
            occupation = self.model.grid.get_cell_list_contents([new_position])
            if len(occupation) < 1:
                self.model.grid.move_agent(self, new_position)
                self.A_r[dir] += R1
                self.A_p[dir] += 1
                #print("Agent",self.unique_id,angledegrees,"Explored-Success")
            else:
                self.A_r[angle] += R2
                self.A_p[angle] += 1
                self.collision += 1
                #print("Agent",self.unique_id,angledegrees,"Explored-COLLIDE")
        #print(self.A_r)
        self.prevdir = dir

    def step(self):
        self.move()
        print("running")

        
### enter and change parameter values here
init_value = 1 
N = 40
width = 50
height = 50
R1 = 2
R2 = -250
steps = 10000
epsilon_value = 0.45

### this is for setup, don't edit
directionnames = ["North", "West", "East", "South"]
directions = [0,1,2,3] #NWES directions, Von Neumann neighborhood
angles = [0,1,2,3]
init_dir_r = [0,0,0,0]


### this is where the model is initiated and the plot starts

model = CollisionModel(N, width, height, init_value) #this is how many agents there are, w, h and init reward
for i in range(steps): #this number specifies how many runs
    model.step()

### code below plots a nice colorbar where the number of agents in the field can be seen
#agent_counts = np.zeros((model.grid.width, model.grid.height))
#for cell in model.grid.coord_iter():
#    cell_content, x, y = cell
#    agent_count = len(cell_content)
#    agent_counts[x][y] = agent_count
#plt.imshow(agent_counts, interpolation='nearest')
#plt.colorbar()

### code below plots the reward for one agent, say agent 14
#one_agent_reward = agent_reward.xs(14, level="AgentID")
#one_agent_reward.reward.plot()

### code below prints and plots the average reward accross agents
#average_reward = model.datacollector.get_model_vars_dataframe()
#print(average_reward)
#average_reward.plot()
#plt.show()

### code below prints and plots the average number of (would-be) collisions accross agents
#average_collision = model.datacollector.get_model_vars_dataframe() -1 #-1 is necessary because start is at 1
#print(average_collision)
#average_collision.plot()
#plt.show()

### code to plot collisions per angle
average_reward = model.datacollector.get_model_vars_dataframe()
average_reward.plot()
plt.ylabel("Average Reward")
plt.xlabel("Steps") 
plt.savefig('saved_plots/N={0}_R1={1}_R2={2}_epsilon={3}_steps={4}_width={5}_height={6}.png'.format(N, R1, R2, epsilon_value, steps, width, height))
plt.show()

