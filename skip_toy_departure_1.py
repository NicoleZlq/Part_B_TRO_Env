import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import copy

from gymnasium.envs.registration import register

import pass_process

from gymnasium.experimental.vector import VectorEnv
class SkipToy1(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    STOP = 1
    SKIP= 0

    def __init__(self, num_trains=6, num_stations=6, num_time =60,  pass_od = None, render_mode="console"):
        super(SkipToy1, self).__init__()
        self.render_mode = render_mode
        #passengers flow, dict
        self.OriginalOD = pass_od.copy()

        # numbers of trains
        self.train_num = num_trains
        # number of station
        self.station_num = num_stations
        # peak hour time (1 time inverval = 1 min)
        self.num_time = num_time
        
        
        # the dwell time is also the action list, min
        self.dwell_time = [20, 25, 35, 45, 50, 55, 60, 65,70]

        self.running_time =100

        self.train_cap = 250
        
        self.length_info = 50
        
        self.onboard_dict = {key: [] for key in range(0, self.train_num)}
        

        self.boarding_speed = 10
        #c_0
        self.discount_boarding_speed = 5
        #minimum departure-arrive \theta_st
        self.min_da = 25
        
        #minimum arrive-arrive \theta_se 
        self.min_aa = 30
        
        #minimum departure-departure \theta
        self.min_dd = 30
        
        #departure interval from the first station
        self.departure_interval = 120
        #use to calculate the number of missing train
        self.missing_interval = self.departure_interval + self.dwell_time[len(self.dwell_time) // 2 ]
        
        

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = len(self.dwell_time)
        self.action_space = spaces.Discrete(n_actions)
        # initial state
        state_length = 1+1+self.station_num+self.length_info+self.length_info
        
        self.ini_state = [0] * state_length
        
        max_value = [1000] * state_length
        
        self.observation_space = spaces.Box(
        low=np.array(self.ini_state), high=np.array(max_value), shape=(state_length,), dtype=np.int64
    )

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        self.train_index = 0

        self.station_index = 0


        self.departure_recode = {key: [None] * self.station_num for key in range(self.train_num)}
        
        

        self.arrive_recode = {key: [None] * self.station_num for key in range(self.train_num)}

        self.StartDepartTime = 0

        self.PassengerTotal = {key: [] for key in range(0, self.station_num )}
        

        self.TotalWaitingTime = []

        self.TotalTravelingTime = []
        #missing train
        self.MissingTrain =[]
        
        self.departure_recode[self.train_index][self.station_index] = 0
        self.arrive_recode[self.train_index][self.station_index] = 0
        
        self.max_added_reward, self.max_reward, self.max_penatly = 0,0,0

        
        
        self.PassOD = copy.deepcopy(self.OriginalOD)        
        
        
        self.FirstBoardProcess()

        return np.array(self.ini_state).astype(np.float32), {} 

    def step(self, action):
        
        self.action = self.dwell_time[action] 
        
        if self.station_index == self.station_num-1:
        
            penatly = self.UpdateTrain()

        if self.station_index !=0 :

        #update arrvie time
            self.arrive_recode[self.train_index][self.station_index]  = self.departure_recode[self.train_index][self.station_index-1] + self.running_time
            
            self.PassengerArriveProcess() 
            
            penatly = self.ActionDetection()
            
            if self.action:

                self.PassengerBoardProcess() 


        terminated = bool((self.station_index == self.station_num-2 and self.train_index == self.train_num-1) or self.action is None)

        reward = -sum([(num // self.missing_interval) ** 2 for num in self.TotalWaitingTime])/(len(self.TotalWaitingTime))+ penatly*2
        
        self.station_index += 1

    
        print('*************', terminated)
        if terminated:
            
   
            if self.action is not None:

                added_reward = -self.CalculateFinalReward()/(len(self.TotalWaitingTime)) 
                reward += added_reward
                
            self.MissingTrain = [(num // self.missing_interval)  for num in self.TotalWaitingTime]


        truncated = False   

        print('reward:', reward)


        info = {'missing':self.MissingTrain, 'traveling':self.TotalTravelingTime}
        
        state = self.UpdateState()

        return (
            np.array(state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        pass

    def close(self):
        pass
    
    def UpdateState(self):

        info = [self.train_index, self.station_index]
        
        remaining_pass =  [len(value_list) for value_list in self.PassengerTotal.values()]
        
        arriving_time = [value_list[0] for value_list in self.onboard_dict[self.train_index][:self.length_info]]

        arriving_time.extend([0] * (self.length_info - len(arriving_time)))
        
        end_station = [value_list[1] for value_list in self.onboard_dict[self.train_index][:self.length_info]]

        end_station.extend([0] * (self.length_info - len(end_station)))

        
        return  info + remaining_pass + arriving_time + end_station 
    
    
    def ActionDetection(self):
        
        new_action = self.action
        
        penatly = 0
        
        departure_time = self.arrive_recode[self.train_index][self.station_index] + self.action
        
        if self.train_index != 0:
            

            arrive_time_next_station = departure_time + self.running_time

            
            diff_ad = arrive_time_next_station - self.departure_recode[self.train_index-1][self.station_index+1]

            diff_aa = arrive_time_next_station - self.arrive_recode[self.train_index-1][self.station_index+1]

            diff_dd = departure_time -self.departure_recode[self.train_index-1][self.station_index]
            
            min_time = min(diff_aa, diff_ad, diff_dd)
            
            if min_time < self.min_aa:
                
                diff_time = self.min_aa - min_time + self.action
                
                new_action = self.FindNewAction(self.dwell_time, diff_time)
                
                if new_action in self.dwell_time:
                
                    penatly = (self.action -new_action) * self.action/1e5
                
                    departure_time = self.arrive_recode[self.train_index][self.station_index] + new_action
                    
                    self.action = new_action
                else:
                    self.action = None
                    count_none = sum(1 for sublist in self.departure_recode.values() for value in sublist if value is None)
                    penatly = -count_none 
                
        self.departure_recode[self.train_index][self.station_index] = departure_time
         
        
        return  penatly
    
    
    def FindNewAction(self,numbers, target):

        closest_number = None
        min_difference = float('inf')

        for number in numbers:
            if number >= target:
                difference = number - target
                if difference < min_difference:
                    min_difference = difference
                    closest_number = number

        return closest_number
    
    
    def FirstBoardProcess(self):

        
        departure_time = self.train_index * self.departure_interval
        

        
        if self.arrive_recode[self.train_index][self.station_index] in self.PassOD.keys():
            values = self.PassOD[departure_time]  
            boarding = [value for value in values if value[0] == self.station_index]
            updated_values = [value for value in values if value[0] != self.station_index]
            self.onboard_dict[self.train_index] = boarding
            self.PassOD[departure_time] =  updated_values
            
        self.station_index+=1
        
        self.arrive_recode[self.train_index][self.station_index] = self.running_time
            
        
                    
    def PassengerArriveProcess(self):

            
        time_interval = [0, self.arrive_recode[self.train_index][self.station_index]]

        
        for key in self.PassOD:
            if time_interval[0] <= key <= time_interval[1]:
                while len(self.PassOD[key])>0:
                    value = self.PassOD[key].pop(0)     
                    new_tuple = value[:0] + (key,) + value[1:]  
                    self.PassengerTotal[value[0]].append(new_tuple)
                    
                    
        for key in self.PassengerTotal:
                    
            self.PassengerTotal[key] = sorted(self.PassengerTotal[key], key=lambda x: x[0])

    
    def PassengerBoardProcess(self):
        
        #Alighting Process
        ArrivingTime = []
        WaitingTime = []
        
        Passengers = self.onboard_dict[self.train_index]
        
        subdict = [value for value in Passengers if value[1] == self.station_index]
        
        for sublist in subdict:
            self.TotalTravelingTime.append(self.arrive_recode[self.train_index][self.station_index]- sublist[0] )
            

        self.onboard_dict[self.train_index] = [value for value in Passengers if value[1] > self.station_index]
        
        #calcualte alighting time  and boarding time    
        alighting_num = len(Passengers) - len(self.onboard_dict[self.train_index])
        alighting_time = alighting_num / self.boarding_speed
        
        boarding_time = self.action - self.discount_boarding_speed - alighting_time
        
        #the maximum boarding passenger
    
        max_boarding_passenger = self.boarding_speed * boarding_time
        
        # print('max boarding passengers:', max_boarding_passenger)
        
        # print('before')
        # print('waiging passengers:',len(self.PassengerTotal[self.station_index]))
        # print('numbers of onboard passenger', len(self.onboard_dict[self.train_index])) 
        
        boarding_num = 0
                  
       #Boarding Process
        while len(self.onboard_dict[self.train_index]) < self.train_cap and len(self.PassengerTotal[self.station_index]) >0 and boarding_num < max_boarding_passenger:   #constrain: train capacity
            boarding_num +=1
            boarding_passenger = self.PassengerTotal[self.station_index].pop(0)  
            #the all feasible passenger has beed boarded
            if boarding_passenger[0] > self.arrive_recode[self.train_index][self.station_index]:
                self.PassengerTotal[self.station_index].insert(0, boarding_passenger)
                break
            ArrivingTime.append(boarding_passenger[0])
            self.onboard_dict[self.train_index].append(boarding_passenger)
        
            
        #calcualte the waiting time until the passenger board on the train
    #    ArrivingTime = [value[0] for value in self.onboard_dict[self.train_index]]
        WaitingTime = [self.arrive_recode[self.train_index][self.station_index] - item for item in ArrivingTime]

        
        self.TotalWaitingTime.extend(WaitingTime)
        
        # print('after')
        # print('numbers of remaining passenger', len(self.PassengerTotal[self.station_index]))
        # print('numbers of onboard passenger', len(self.onboard_dict[self.train_index]))
        

            
    def UpdateTrain(self):

            
        self.arrive_recode[self.train_index][self.station_index] = self.departure_recode[self.train_index][self.station_index-1] + self.running_time
            
        self.departure_recode[self.train_index][self.station_index] = self.arrive_recode[self.train_index][self.station_index]
            
        self.TotalTravelingTime.extend([self.arrive_recode[self.train_index][self.station_index]- item[0]  for item in  self.onboard_dict[self.train_index]])

        self.onboard_dict[self.train_index] = []    #in the end station, all of passenger should be alighting
        

        
        self.train_index +=1
        self.station_index = 0
        
        self.arrive_recode[self.train_index][self.station_index] =  self.arrive_recode[self.train_index-1][self.station_index]
        
        penatly = self.ActionDetection()
        
        if self.action:
             
    
            self.arrive_recode[self.train_index][self.station_index] = self.departure_recode[self.train_index][self.station_index]  = self.arrive_recode[self.train_index-1][self.station_index] + self.action

        
            passengers = self.PassengerTotal[self.station_index]
            
            selected_passenger = [value for value in passengers if value[0] <= self.arrive_recode[self.train_index][self.station_index]]
            updated_passengers = [value for value in passengers if value[0] > self.arrive_recode[self.train_index][self.station_index]]
            
            
            num_to_passenger =  min(len(selected_passenger), self.train_cap)
            
            self.onboard_dict[self.train_index] =  selected_passenger[:num_to_passenger]
            
            updated_passengers += selected_passenger[num_to_passenger:]
            
            
            self.PassengerTotal[self.station_index] = updated_passengers
            
            WaitingTime = [self.arrive_recode[self.train_index][self.station_index] - item[0] for item in self.onboard_dict[self.train_index]]


            self.TotalWaitingTime.extend(WaitingTime)
        
        # print('in the first staition')
        # print('waiging passengers:',len(self.PassengerTotal[self.station_index]))
        # print('numbers of onboard passengers:',len(self.onboard_dict[self.train_index]))
        
        return penatly

    def CalculateFinalReward(self):
        sum_of_squares = 0
        
        self.arrive_recode[self.train_index][self.station_index] = self.departure_recode[self.train_index][self.station_index-1] + self.running_time

        # Iterate through the dictionary values
        for key, value in self.PassengerTotal.items():

            if value:  


                for i in value:
                    first_item = i[0]
                 
                    sum_of_squares += ((self.arrive_recode[self.train_index][self.station_index]- first_item) //  self.missing_interval) ** 2
                    
                    WaitingTime = [self.arrive_recode[self.train_index][self.station_index] - first_item]

                    self.TotalWaitingTime.extend(WaitingTime)
                    
        return sum_of_squares
    
    
    def skip_constrain(self):
        pass
        

            

            

            
           
            

    

    
    
      

      
