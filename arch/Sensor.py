import numpy as np
from typing import List, Tuple
from .Central_processor import Central_processor
class Sensor:
    """
    A class representing a sensor.

    Attributes:
    - memory_table (Dict): A dictionary to store sensor memory.

    Sensor related parameters:
    - m (List[int]): A list of indices of state components sensor j measures.
    - tau (float): Sampling period, defaults to tau_j = 5.0 for all sensors.
    - b (float): Sensor offsets.
    - H (int): Memory table size of sensor j, defaults to H = 1.
    - omega_sq (int): Sensor White noise variance, defaults to omega_sq = 0.2.
    - epsilon (float): Triggering rule threshold value, defaults to epsilon = 0.5.

    Methods:
    - __init__: Initializes a new instance of the Sensor class.
    - set_cpu: Associates a CPU with the sensor.
    - run: Executes the sensor process and determines whether data needs to be transmitted to the CPU.
    - sampling_rule: Samples the environment for components observed by the sensor.
    - triggering_rule: Determines if the triggering rule is satisfied for transmitting data to the CPU.
    - triggering_rule_fb: A variation of the triggering rule (Feedback) for transmitting data to the CPU.
    - representation_rule: Produces the sensor's representation based on the memory table.
    - send_data_to_cpu: Sends the sensor representation to the CPU.
    - recieve_data_from_cpu: Receives data from the CPU and updates the sensor's memory table.


    Example:
    sensor = Sensor(id=0, m=[0,1], n=2, b=1.0, type='fb')
    """

    def __init__(self,id: int, m: List[int],n :float, b: float, tau: float = 5.0, H: int = 1, omega_sq: float = 0.2, epsilon: float = 0.5, type: str = 'nf') -> None:
        """
        Initializes a new instance of the Sensor class.

        Parameters:
        - id (int): Sensor number.
        - m (List[int]): List of indices of state component sensor j measures.
        - n (int): Length of the state vector.
        - b (float): Sensor offsets.
        - tau (float, optional): Sampling period. Defaults to 5.0.
        - H (int, optional): Memory table size of sensor j. Defaults to 1.
        - omega_sq (float, optional): Sensor White noise variance. Defaults to 0.2.
        - epsilon (float, optional): Triggering rule threshold value. Defaults to 0.5.
        - type (str, optional): Type of sensor. Defaults to 'nf'.

        Returns:
        - None
        """
        self.id = id
        self.m = m
        self.b = b
        self.H = H+1 # need to store two states, current and the previous
        self.tau = tau
        self.type = type
        self.epsilon = epsilon
        self.memory_table: np.ndarray = np.empty((self.H,len(self.m)))
        self.current_index: int = 0
        self.a_star = -1
        self.last_sample_t = [-1]*len(self.m)
        self.cpu =None
    

    def set_cpu(self, cpu: 'Central_processor') -> None:
        """
        Associates a CPU (Central Processor) with the sensor.

        Parameters:
        - cpu (Central_processor): The Central Processor object to associate with the sensor.

        Returns:
        - None
        """
        self.cpu = cpu
        

    def run(self, i: int, new_x: np.ndarray) -> Tuple[bool, int, List[int], np.ndarray]:
        """
        Executes the sensor process and determines whether data needs to be transmitted to the CPU.

        Parameters:
        - i (int): Time index.
        - new_x (np.ndarray): The new environment input.

        Returns:
        - Tuple[bool, int, List[int], np.ndarray]: A boolean flag indicating if data should be transmitted,
        the sensor ID, transmitting indices, and the data to be sent.
        """
         
        self.a_star = int(np.floor(((i-self.b)/self.tau)))
        # print(f"a_star: ", self.a_star)
        self.sampling_rule(new_x)
        self.last_sample_t = [i]*len(self.m)
        # get state reperesentation
        y_hat = self.representation_rule()
        if self.type == 'nf':
            triggered = self.triggering_rule(self.a_star)
        else:
            triggered = self.triggering_rule_fb(self.a_star)

        #print(f"id: {self.id}, type: {self.type}, y_hat: {y_hat}, current sample: {self.memory_table[self.current_index]}, idx: {self.current_index}")
        # if triggered:
        #     self.send_data_to_cpu(t=i, y_hat=y_hat)
        idx_transmitting = [m for m, condition in zip(self.m, triggered) if condition]
        y_hat = y_hat[triggered].copy()
        triggered = any(value for value in triggered)
        return triggered, self.id, idx_transmitting, y_hat

    def sampling_rule(self, new_x: np.ndarray) -> None:
        """
        Samples the environment for components observed by the sensor and stores the data in the memory table.

        Parameters:
        - new_x (np.ndarray): The new environment input.

        Returns:
        - None
        """
        
        y = new_x[self.m].copy()
        self.current_index = self.a_star % self.H
        ###print("mem idx: ", self.current_index,"a*: ", self.a_star,"mem_table_size: ", self.H)
        #print(f"({self.id})mem: ", self.memory_table)
        self.memory_table[self.current_index] = y

    def triggering_rule(self, a_star: int) -> List[bool]:
        """
        Determines if the triggering rule (T3) is satisfied for transmitting data to the CPU.

        Parameters:
        - a_star (int): The number of sampled state -1 by the sensor.

        Returns:
        - List[bool]: A boolean flag for each component indicating if the sensor needs to transmit data to the CPU.
        """

        upper_bound = min(a_star+1,self.H)
        if(upper_bound == 0 or a_star==0):
            return  [True]*len(self.m)
        
        trigger = [False]*len(self.m)
        # print(f"triggering rule; {a_star, upper_bound}")
        for j in range(len(self.m)):
            sum_difference = 0
            for i in range(1,upper_bound):
                # print("nope")
                diff = self.memory_table[self.current_index][j] - self.memory_table[(self.current_index-i)%self.H][j]
                sum_difference += np.abs(diff)
                # print(f"triggering rule:({self.current_index}, {(self.current_index-1)%self.H}) {self.memory_table[self.current_index]}, {self.memory_table[(self.current_index-1)%self.H]}")

            average_difference = sum_difference/upper_bound
            # print("avg_diff: ", average_difference)
            if(average_difference>=self.epsilon):
                trigger[j] = True
        return trigger

    def triggering_rule_fb(self, a_star) -> List[bool]:
        """
        A variation of the triggering rule (T3)(Feedback) for transmitting data to the CPU.

        Parameters:
        - a_star (int): The number of sampled state -1 by the sensor.

        Returns:
        - List[bool]: A boolean flag for each component indicating if the sensor needs to transmit data to the CPU.
        """
        upper_bound = min(a_star+1,self.H)
        if(upper_bound == 0 or a_star==0):
            return [True]*len(self.m)
        
        trigger = [False]*len(self.m)
        # print(f"triggering rule; {a_star, upper_bound}")
        for j in range(len(self.m)):
            sum_difference = 0
            for i in range(1,upper_bound):
                # print("nope")
                diff = self.memory_table[self.current_index][j] - self.memory_table[(self.current_index-i)%self.H][j]
                sum_difference += np.abs(diff)
                # print(f"triggering rule:({self.current_index}, {(self.current_index-1)%self.H}) {self.memory_table[self.current_index]}, {self.memory_table[(self.current_index-1)%self.H]}")

            average_difference = sum_difference/upper_bound
            # print("avg_diff: ", average_difference)
            if(average_difference>=self.epsilon):
                trigger[j] = True

        return trigger



    def representation_rule(self) -> np.ndarray:
        """
        Produces the sensor's representation (y_hat) based on the memory table.

        Returns:
        - np.ndarray: The sensor's representation of the memory table.
        """
        y_hat = self.memory_table[self.current_index]
        return y_hat
    
    # def send_data_to_cpu(self, t: int, y_hat: np.ndarray)->None:
    #     """
    #     Sends the sensor representation (y_hat) to the CPU.

    #     Parameters:
    #     - t (int): The current time.
    #     - y_hat (np.ndarray): The sensor representation to be transmitted.

    #     Returns:
    #     - None.
    #     """
    #     self.cpu.recieve_data(id=self.id,t=t,y_hat=y_hat)

    def recieve_data_from_cpu(self,t_sensor_start: np.ndarray, x_hat: np.ndarray, idx_trans: List[int])->None:
        """
        Receives data from the CPU and updates the sensor's memory table.

        This method behaves like a sampling rule, updating the memory table with new sensor data received from the CPU.

        Parameters:
        - t_sensor_start (np.ndarray): Array containing the transmission time by a sensor of data received from the CPU. 
                                    for each index.
        - x_hat (np.ndarray): Array of sensor data received from the CPU.
        = idx_trans (List[int]): List of indices indicating which sensor data is being transmitted.

        Returns:
        - None.
        """
        # can actually treat it as a sampling rule, where it updates the memory table. 
        # print(f"received data, {self.id}, x: {x_hat}")
        
        # print(self.memory_table)
        for i, m in enumerate(idx_trans):
            idx_m = self.m.index(m)
            # the sensor's last reading is more recent then ignore
            if(self.last_sample_t[idx_m] >= t_sensor_start[i]):
                continue
            #otherwise we need to update the current state of sensor
            self.memory_table[self.current_index][idx_m] = x_hat[i]
            self.last_sample_t[idx_m] = t_sensor_start[i]


     