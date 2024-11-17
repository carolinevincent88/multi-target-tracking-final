import random
#random.seed(123)
from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box

AGENTS = 5
CRASHREWARD = -20
print("CR:", CRASHREWARD)
GOALREWARD = 100
LEFTWALL = 0
RIGHTWALL = 15
BOTTOMWALL = 0
TOPWALL = 15
MAXGRIDCELLX = TOPWALL-2
MAXGRIDCELLY = RIGHTWALL-2
ACTIONSIZE = 5
MAXTIMESTEPS = 5000
IDLEPENALTY = 0
TIMESTEPPENALTY = -1
WRONGDIRECTIONPENALTY = 0
RIGHTDIRECTIONREWARD = 0

obstacles = []

if AGENTS==2:
    goal_locations = [(3, 0), (6, 0)]
elif AGENTS==3:
    goal_locations = [(3, 0), (6, 0), (9, 0)]
elif AGENTS==4:
    goal_locations = [(3, 0), (6, 0), (9, 0), (12, 0)]
elif AGENTS==5:
    goal_locations = [(3, 0), (6, 0), (9, 0), (12, 0), (0, 0)]

ymin = 0

#define the observation space
obs_low = np.array([-MAXGRIDCELLX, -MAXGRIDCELLY] * AGENTS, dtype=np.int32)
obs_high = np.array([MAXGRIDCELLX, MAXGRIDCELLY] * AGENTS, dtype=np.int32)
obs_space = Box(low=obs_low, high=obs_high, dtype=np.int32)

#create an Obstacle class for static obstacles 
class Obstacle():
    def __init__(self, location):
        #initialize the x and y location variable of the obstacle
        self.x = location[0]
        self.y = location[1]

# a function that places obstacles 
def place_obstacles(drone_list, random_locations):
    #create list of obstacles
    list_of_obstacles = []
    locations = []
    for count in random_locations:
        list_of_obstacles.append(Obstacle(count))
    for obs in list_of_obstacles:
        locations.append((obs.x, obs.y))
    return list_of_obstacles

#create a drone class 
class Drone():
    def __init__(self):
        #initialize the x and y location variable of the drone
        self.x = None
        self.y = None
        #initialize the x and y location variable of the goal
        self.goalx = None
        self.goaly = None
        #initialize the at_goal variable to false
        self.at_goal = False
        #initialize the rewards variable to 0
        self.rewards = 0

    #a function that checks if the drone is at the same location as other drones or obstacles 
    def check_drone_collisions(self, drones):
        for drone in drones:
            if drone is self:
                continue
            else:
                #check if these two drones are colliding
                if drone.x == self.x and drone.y == self.y:
                    return True
        return False
    
    #a function that checks if the drone is at the same location as other drones or obstacles 
    def check_obstacle_collisions(self, drone, obstacles):
        for obstacle in obstacles:
            #check if drone crashed into obstacle 
            if drone.x == obstacle.x and drone.y == obstacle.y:
                return True
        return False
    
    #a function that checks if all drones have reached their goals
    def check_goals(self, drones):
        #check if each drone is at its goal; return true if all drones are at their goal; else, return false
        for drone in drones:
            if drone.x != drone.goalx or drone.y != drone.goaly:
                return False
        return True

class CustomEnvironment(gym.Env):
    def __init__(self, epsilon=1.0):
        super(CustomEnvironment, self).__init__()
        #define the observation space
        self.observation_space = obs_space
        if AGENTS == 1:
            self.action_space = Discrete (ACTIONSIZE)
        else:
            self.action_space = MultiDiscrete ([ACTIONSIZE]*AGENTS)
 
        #create a list of drones that will be used in the environment
        self.original_drones = []
        #add the number of drones specified to the list
        for agent in range(AGENTS):
            self.original_drones.append(Drone())

        #initialize the timestep variable
        self.timestep = None
        #initialize the total rewards
        self.total_rewards = 0
        #initialize the reached goals variable
        self.reached_goals = False
        #initialize a total system crash count
        self.crash_count = 0

        self.obstacles = []

        self.metadata = {}

    def reset(self, randomReset=True, scenario=1):
        #reset list of drones
        self.drones = deepcopy(self.original_drones)

        #set the timestep to 0
        self.timestep = 0
        #set the total rewards to 0
        self.total_rewards = 0
        #reset the crash count
        self.crash_count = 0
        #set the reached goals variable to false
        self.reached_goals = False
        #reset drones
        for drone in self.drones:
            #set each drone's rewards to 0
            drone.rewards = 0
            #set each drone's at_goal variable to false
            drone.at_goal = False

        
        drone_index = 0
        for drone in self.drones:
            drone.goalx = goal_locations[drone_index][0]
            drone.goaly = goal_locations[drone_index][1]
            drone_index += 1

        #set the drones in a random location within the grid size of BOTTOMWALL x TOPWALL where they can't be on the same spot or on the same spot as the goal
        taken_locations = []
        #add the goal locations to the taken_locations list
        for goal_location in goal_locations:
            taken_locations.append(goal_location)
        #set the drones in a random location within the grid size of BOTTOMWALL x TOPWALL where they can't be on the same spot or on the same spot as the goal
        for drone in self.drones:
            if randomReset:
                drone_location = (random.randint(0,RIGHTWALL-1), random.randint(ymin,RIGHTWALL-1))
                while drone_location in taken_locations:
                    drone_location = (random.randint(0,RIGHTWALL-1), random.randint(ymin,RIGHTWALL-1))
                taken_locations.append(drone_location)
                drone.x = drone_location[0]
                drone.y = drone_location[1]
            else:
                print('error, should be random resetting in training')
        
        #for drone in self.drones, add observation
        observations = []
        for drone in self.drones:
            observations.extend([drone.x-drone.goalx, drone.y-drone.goaly])
       
        #return the observations as a NP array
        observations = np.array(observations, dtype=np.int32).flatten()
        return observations



    def step(self, actions):
        #reset each drone's rewards to 0
        for drone in self.drones:
            drone.rewards = 0
        if np.isscalar(actions):
            actions = np.array([actions])
        # initialize actionNumber to 0 to keep track of which drone's action is being executed in the list of actions 
        actionNumber = 0
        # Execute actions
        for drone,action in zip(self.drones,actions):
            drone_action = action
            if drone_action == 0: #left
                if drone.x >LEFTWALL:
                    #give reward if the drone is moving towards its goal
                    if drone.x > drone.goalx:
                        drone.rewards += RIGHTDIRECTIONREWARD
                        #print("left - reward")
                    #give penalty if the drone is moving away from its goal
                    else:
                        drone.rewards += WRONGDIRECTIONPENALTY
                        #print("left - penalty")
                    #move the drone to the left
                    drone.x -= 1
                else: #penalize for not moving 
                    drone.rewards += IDLEPENALTY
                    #print("stay - penalty")
            elif drone_action == 1: #right
                if drone.x <RIGHTWALL-1:
                    #give reward if the drone is moving towards its goal
                    if drone.x < drone.goalx:
                        drone.rewards += RIGHTDIRECTIONREWARD
                        #print("right - reward")
                    #give penalty if the drone is moving away from its goal
                    else:
                        drone.rewards += WRONGDIRECTIONPENALTY
                        #print("right - penalty")
                    #move the drone to the right
                    drone.x += 1
                else: #penalize for not moving
                    drone.rewards += IDLEPENALTY
                    #print("stay - penalty")
            elif drone_action == 2: #down
                if drone.y >BOTTOMWALL:
                    #give reward if the drone is moving towards its goal
                    if drone.y > drone.goaly:
                        drone.rewards += RIGHTDIRECTIONREWARD
                        #print("down - reward")
                    #give penalty if the drone is moving away from its goal
                    else:
                        drone.rewards += WRONGDIRECTIONPENALTY
                        #print("down - penalty")
                    #move the drone down
                    drone.y -= 1
                else: #penalize for not moving
                    drone.rewards += IDLEPENALTY
                    #print("stay- penalty")
            elif drone_action == 3: #up
                if drone.y <TOPWALL-1: 
                    #give reward if the drone is moving towards its goal
                    if drone.y < drone.goaly:
                        drone.rewards += RIGHTDIRECTIONREWARD
                        #print("up - reward")
                    #give penalty if the drone is moving away from its goal
                    else:
                        drone.rewards += WRONGDIRECTIONPENALTY
                        #print("up - penalty")
                    #move the drone up
                    drone.y += 1
                else: #penalize for not moving
                    drone.rewards += IDLEPENALTY
                    #print("stay - penalty")
            elif drone_action == 4: #stay
                #penalize for staying if the drone is not at its goal
                if drone.at_goal == False:
                    drone.rewards += IDLEPENALTY
                    #print("stay - penalty")
            #increment actionNumber by 1 to move on to the next drone's action
            actionNumber+=1

            
        
        ## Check termination conditions
        termination = False

        # Check if all drones have reached their goals
        if drone.check_goals(self.drones):
            termination = True
            self.reached_goals = True
            #reward each drone that has reached its goal by adding GOALREWARD to drone.rewards
            for drone in self.drones:
                if drone.at_goal == False:
                    drone.rewards += GOALREWARD
                    drone.at_goal = True
        else:
            # Check if any drones have crashed into other drones or obstacles 
            for drone in self.drones:
                #check if two drones crashed into each other
                if drone.check_drone_collisions(self.drones):
                    #penalize for crashing
                    drone.rewards += CRASHREWARD 
                    self.crash_count += 1
                #check to see if drone is at goal
                else:
                    if drone.x == drone.goalx and drone.y == drone.goaly:
                        #if the drone was not there previously, reward for reaching goal
                        if drone.at_goal == False:
                            drone.rewards += GOALREWARD
                            #set drone's at_goal variable to true
                            drone.at_goal = True
                    else:
                        #if drone at_goal was true, penalize and set at_goal to false
                        if drone.at_goal == True:
                            drone.rewards -= GOALREWARD
                            drone.at_goal = False
                #check if any drones crashed into obstacls 
                if drone.check_obstacle_collisions(drone, self.obstacles):
                    #penalize for crashing
                    drone.rewards += CRASHREWARD 
                    self.crash_count += 1

        for drone in self.drones:
            if drone.at_goal == False:
                drone.rewards += TIMESTEPPENALTY

        
        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > MAXTIMESTEPS:
            termination = True
        else:
            self.timestep += 1


        # Get observations
        observations = []
        for drone in self.drones:
            observations.extend([drone.x-drone.goalx, drone.y-drone.goaly])
        #add obstacles
        '''obstacle_obs = []
        for obstacle in self.obstacles:
            obstacle_obs.extend([obstacle.x, obstacle.y])
        observations.extend(obstacle_obs)
        '''
        #make the observations a NP array
        observations = np.array(observations, dtype=np.int32).flatten()



        #make a list of rewards for each drone
        rewards = []
        for drone in self.drones:
            rewards.append(drone.rewards)
        
        #sum the rewards for training
        training_rewards = sum(rewards)

        info = {'reached_goals': self.reached_goals, 'crash_count': self.crash_count}

        return observations, training_rewards, termination, info

    def render(self):
        grid = np.zeros((RIGHTWALL, TOPWALL), dtype=int)
        low_number = 1
        high_number = 9
        obstacle_number = 4
        for drone in self.drones:
            grid[((TOPWALL-1)-drone.y), drone.x] = low_number
            grid[((TOPWALL-1)-drone.goaly), drone.goalx] = high_number
            high_number -= 1
            low_number += 1


        print(f"{grid} \n")
        for drone in self.drones:
            #print drone_at_goal
            print(f"Drone at goal: {drone.at_goal}")

    def close(self):
        pass

    def seed(self, seed=None):
        pass
