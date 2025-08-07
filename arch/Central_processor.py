import pandas as pd
import numpy as np
from typing import List, Tuple
from .Sensor import Sensor

class Central_processor:
    """
    A class representing a central processor.
    Attributes:
    - memory_table (Dict): A dictionary to store CPU memory.

    CPU Related Parameters:
    - P (List[float]): [PU, PD] denotes the power expenditure cost incurred by sensor j in transmitting (PU) and receiving (PD) one packet. Defaults to [5.0, 3.0].
    - t_d (List[float]): Communication delays for uplink and downlink. Assumes uplink delay (t_du) is greater than or equal to downlink delay (t_dd). Defaults to [2.0, 1.0].
    - pb (float): Broadcast probability within the range (0, 1]. Default value is assumed to be within this range.
    - Hc (int): Central processor memory table. Default value is 1.

    Methods:
    - __init__: Initializes a new instance of the Central_processor class.
    - fusion_rule: Implements the fusion rule to update x_hat with the most recent data from sensors.
    - broadcast_rule: Determines whether to broadcast data based on specified rules.
    - recieve_data: Receives data from sensors and updates the central processor's memory.
    - send_data_to_all_sensors: Sends updated data to all associated sensors.
    """

    def __init__(self,n: int, x0: np.ndarray, sensors: List[Sensor], P: List[float] = [5.0, 3.0], t_d: List[float] = [2.0, 1.0], pb: float = 0.5, Hc: int = 1, type: str = 'nf') -> None:
        """
        Initializes a new instance of the Central_processor class.

        Parameters:
        - sensors (List[Sensor]): List of sensor objects.
        - P (List[float], optional): Power expenditure costs. Defaults to [5.0, 3.0].
        - t_d (List[float], optional): Communication delays [uplink, downlink]. Defaults to [2.0, 1.0].
        - pb (float, optional): Broadcast probability. Defaults to 0.5.
        - Hc (int, optional): Central processor memory table. Defaults to 1.
        - type (str, optional): Type of processor. Defaults to 'nf'.

        Returns:
        - None
        """
        self.H = Hc
        self.sensors = sensors
        self.M = len(sensors)
        self.n = n
        self.memory_table: np.ndarray = np.empty((self.H,n))
        self.memory_table_idx = 0
        self.V = pd.DataFrame({'t_start': [-1] * self.M, 't': [-1] * self.M, 'y_hat': np.full((self.M, n), np.nan).tolist()})
        self.x_hat = np.full(n, np.nan)
        self.x_time_sampled = np.full(n, -1)
        self.type = type
        self.t_d_u = t_d[0]  # uplink delay, 
        self.t_d_d = t_d[1] #  downlink dealy
        self.pb = pb

        # for graphing later
        self.x = pd.DataFrame({'t': 0, 'x': [x0]})

        for sensor in sensors:
            self.V.at[sensor.id, 't'] = 0
            self.V.at[sensor.id, 'y_hat'] = x0
            self.V.at[sensor.id, 't_start'] = 0



    def fusion_rule(self, current_time: float) -> None:
        """
        Fusion rule F1 as set up in the paper.
        Componenet-replacement rule.
        Update the x_hat whenever a new component arrives.

        Parameters:
        - current_time (float): Current time.

        Returns:
        - None
        """
        # sort by time, so latest time is at the top
        V_upto_t =  self.V[self.V['t']<=current_time]
        V_sorted =  V_upto_t.sort_values(by='t_start', ascending=False)
        # remove all sensors where we havent received any data
        V_sorted = V_sorted[V_sorted['t'] != -1]
        # extract all recieved data
        y_hat_np_array = np.array(V_sorted['y_hat'].values.tolist())
        # get the most recent data 

        for i in range(self.n):
            y_hat_i = y_hat_np_array[:, i]
            # x_i not received yet
            if(np.all(np.isnan(y_hat_i))):
                continue
            else:
                recent_update_idx = np.where(~np.isnan(y_hat_i))[0][0]
                if( self.x_time_sampled[i]<V_sorted['t_start'].iloc[recent_update_idx]):
                    self.x_hat[i] = y_hat_i[recent_update_idx]
                    self.x_time_sampled[i] = V_sorted['t_start'].iloc[recent_update_idx]
            
        if (self.x['t'] == current_time).any():
            idx =  self.x[self.x['t'] == current_time].index.tolist()[0]
            self.x.at[idx, 'x'] =  self.x_hat.copy()
        else:
            new_data = pd.DataFrame({'t': current_time, 'x': [self.x_hat.copy()]})
            self.x =  pd.concat([self.x, new_data], ignore_index=True)
        

    def broadcast_rule(self) -> np.ndarray:
        """
        Broadcast rule B1 as set up in the paper.
        Instantaneous broadcasting with pb.
        Parameters:
        - None

        Returns:
        - bool: array of [True if broadcasting is allowed for a specific component with prob pb,
                         False otherwise].
        """
        # instaneous broadcast if less than pb

        broadcast = np.random.choice([True, False], size=self.n, p=[self.pb, 1-self.pb])
        
        return broadcast

    def recieve_data(self, id: int, t_start: float, t: float, y_hat: np.ndarray, idx_trans: List) -> Tuple[bool, np.ndarray, np.ndarray, List[float]]:
        """
        Receives data from sensors and updates the central processor's memory.
        Parameters:
        - id (int): Sensor ID.
        - t_start (float): Start time of data transmission.
        - t (float): Current time.
        - y_hat (np.ndarray): Sensor data received.
        - idx_trans (List): List of indices of transmitted data.

        Returns:
        - Tuple[bool, np.ndarray, np.ndarray, List[float]]: A tuple containing information about broadcast, updated data,
                                                        transmitted indices, and associated time information.
        """
        
        # print("recieved at cpu")

        x_hat  = np.full(self.n, np.nan)
        x_hat[idx_trans] = y_hat
        self.V.at[id, 't'] = t
        self.V.at[id, 'y_hat'] = x_hat
        self.V.at[id, 't_start'] = t_start

        # find xhat based on stuff upto now
        self.fusion_rule(current_time=t)
       
        broadcast = False
        idx_to_broadcast = np.empty(self.n)
        time_sampled = []
        if(self.type == 'fb'):
            broadcast_list= self.broadcast_rule()
            idx_to_broadcast = (np.where(broadcast_list)[0]).tolist()
            x_hat = self.x_hat[idx_to_broadcast]
            time_sampled = self.x_time_sampled[idx_to_broadcast].tolist()
            broadcast = np.any(broadcast_list)

        return broadcast, x_hat, idx_to_broadcast, time_sampled
    

    def send_data_to_all_sensors(self, x_hat: np.ndarray, time_sampled: np.ndarray, idx_trans: List[int]) -> None:
        """
        Sends updated data to all associated sensors.
        Parameters:
        - x_hat (np.ndarray): Updated data to be sent.
        - time_sampled (np.ndarray): Time information related to the sent data.
        - idx_trans (List[int]): List of indices indicating transmitted data.

        Returns:
        - None

        """
        for i in range(self.M):
            sensor_obsv_idx = self.sensors[i].m
            # find which idx of this sensor were transmitted
            sensor_idx_trans = list(set(sensor_obsv_idx).intersection(idx_trans))
            
            # if there are some state idx for this sensor which were transmitted
            if len( sensor_idx_trans)!=0:
                sensor_x_hat = np.full(len(sensor_idx_trans), np.nan)
                t_sampled = np.full(len(sensor_idx_trans), -1)
                k=0
                for j, m in enumerate(idx_trans):
                    if m in sensor_obsv_idx:
                        sensor_x_hat[k]=x_hat[j]
                        t_sampled[k]=time_sampled[j]
                        k+=1
                        
                self.sensors[i].recieve_data_from_cpu(t_sensor_start=t_sampled, x_hat=sensor_x_hat, idx_trans=sensor_idx_trans)
