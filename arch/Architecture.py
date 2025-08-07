from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from  .Central_processor import Central_processor
from .Sensor import Sensor


class Architecture:
    """
    A class representing an architecture for simulation.
    Attributes:
    - sensors (List[Sensor]): A list containing Sensor objects.
    - cpu (Central_processor): An instance of the Central_processor class.

    Methods:
    - __init__: Initializes a new instance of the Architecture class.
    - simulate: Simulates the architecture for a given duration.
    - sample_environment: Samples the environment using the sensors.
    - step_environment: Advances the environment state using a random walk.
    - pool_sensors: Updates sensors with environment data and transmits to CPU.
    - send_data_to_cpu_from_buffer: Sends data from sensors to the central CPU.
    - broadcast_data_to_sensor_from_buffer: Broadcasts data from CPU to sensors.
    - plot: Generates plots for communication and state estimation.
    - plot_comm: Plots communication between sensors and CPU over time.
    - plot_states: Plots true and estimated trajectories over time.
    - extend_df: Extends a DataFrame to cover the entire simulation duration.
    - extend_line: Extends the trajectory line to cover the full simulation duration.
    - plot_sensors: Plots communication lines between sensors and CPU.
    - plot_cpu_fb: Plots CPU feedback communication lines.
    - calculate_performance_metrics: Calculates MSE and power expenditure metrics.

    Example:
    architecture = Architecture(x0, n=1, del_T=1, P=1, Tsim=10, omega_sq=1, p=1, epsilon=0.1, tau=0.01, b=1, type='nf')
    """

    def __init__(self, x0: np.ndarray, n: int = 2, del_T: float = 25.0, d: List[float] = [2.0, 4.0], p: float = 0.8, Tsim: int = 200,
                    M: int = 2, m: List[List[int]] = [[0], [0, 1]], tau: List[float] = [5.0, 5.0], b: List[float] = [0, 4.0],
                    H: List[int] = [1, 1], omega_sq: float = 0.2, epsilon: float = 1.5, P: List[float] = [5.0, 3.0],
                    t_d: List[float] = [2.0, 1.0], pb: float = 0.5, Hc: int = 1) -> None:
        """
        Initializes a new instance of the Architecture class.

        Parameters:
        - x0 (np.ndarray): Initial state vector.
        - n (int, optional): State dimension. Defaults to 2.
        - del_T (float, optional): Environment period ∆t∈R+. Defaults to 25.0.
        - d (List[float], optional): Step-size bounds [d_lower, d_upper] . Defaults to [2.0, 4.0].
        - p (float, optional): Non-zero step-size probability p. Defaults to 0.8.
        - Tsim (int, optional): Simulation duration  Tsim ∈ R+. Defaults to 200.
        
        Sensor Parameters:
        - M (int, optional): Number of sensors. Defaults to 2.
        - m (List[List[int]], optional): m_j has a list of indicies of x which sensor j measures. Defaults to [[0], [0, 1]].
        - tau (List[float], optional): Sampling periods for each sensor. Defaults to [5.0, 5.0].
        - b (List[float], optional): Sensor offsets. Defaults to [0, 4.0].
        - H (List[int], optional): Memory table sizes for each sensor, i.e number states each sensor stores. Defaults to [1, 1].
        - omega_sq (float, optional): Sensor White noise variance. Defaults to 0.2.
        - epsilon (float, optional): Triggering rule threshold value. Defaults to 0.5.
        
        CPU Parameters
        - P (List[float], optional): Power expenditure costs incured by sensor j in transmitting/recieving one packet, [PU, PD]. Defaults to [5.0, 3.0].
        - t_d (List[float], optional): Communication delays [uplink, downlink]. Defaults to [2.0, 1.0].
        - pb (float, optional): Broadcast probability. Defaults to 0.5.
        - Hc (int, optional): Central processor memory table size. Defaults to 1.

        Returns:
        - None
        """
        # initialise the true state vec
        self.x: pd.DataFrame = pd.DataFrame({'x': np.zeros((Tsim, n)).tolist(), 'x_fb': np.zeros((Tsim, n)).tolist(), 'x_nf': np.zeros((Tsim, n)).tolist()})
        self.x['x_fb'][0] = self.x['x_nf'][0] = x0

        # environment variables
        self.n: int = n
        self.Tsim: int = Tsim
        self.del_T: float = del_T
        self.d: List[float] = d
        self.M: int = M
        self.omega_sq = omega_sq
        self.td = t_d
        self.p = p
        self.Pu = P[0]
        self.Pd = P[1]

        # setup sensors for both feedback and non-feedback case
        self.nf_sensors: List[Sensor] = []
        self.fb_sensors: List[Sensor] = []
        for j in range(M):
            self.nf_sensors.append(Sensor(id=j, m=m[j], n=n, tau=tau[j], b=b[j], H=H[j], omega_sq=omega_sq, epsilon=epsilon))
            self.fb_sensors.append(Sensor(id=j, m=m[j], n=n, tau=tau[j], b=b[j], H=H[j], omega_sq=omega_sq, epsilon=epsilon, type='fb'))
        
        # setup cpu for both feedback and non-feedback case
        self.cpu: 'Central_processor' = Central_processor(n, x0, self.nf_sensors, P, t_d, pb, Hc)
        self.feedback_cpu: 'Central_processor' = Central_processor(n, x0, self.fb_sensors, P, t_d, pb, Hc, type="fb")
        # set the upward link from sensors to cpu
        self.set_sensors_to_cpu_link(self.nf_sensors, self.cpu)
        self.set_sensors_to_cpu_link(self.fb_sensors, self.feedback_cpu)

        # wait buffer for transmissions from sensors to cpu
        self.buffer_fb = pd.DataFrame()
        self.buffer_nf = pd.DataFrame()

        # wait buffer for broadcasts
        self.broadcast_buffer = pd.DataFrame()


        # broadcast times for graphing
        self.sensor_broadcast_times = pd.DataFrame()
        self.cpu_broadcast_times = pd.DataFrame()

        # values for Power expend calculatios
        self.int_transm_fb = np.zeros(self.M)
        self.int_rec_fb = 0
        self.int_transm_nf = np.zeros(self.M)


    def set_sensors_to_cpu_link(self, sensors:List['Sensor'], cpu:'Central_processor') -> None:
        """
        Associates a list of sensors with a central processing unit (CPU).

        This method links each sensor object in the provided list to the specified CPU.

        Parameters:
        - sensors (List['Sensor']): List of Sensor objects to be associated with the CPU.
        - cpu ('Central_processor'): Central processing unit to which sensors will be linked.

        Returns:
        - None
        """
        for sensor in sensors:
            sensor.set_cpu(cpu)
    
    def simulate(self, Tsim: int) -> None:
        """
        Simulates the architecture for a given duration.

        This method runs the simulation of the architecture for a specified duration.

        Parameters:
        - Tsim (int): The duration of the simulation.

        Returns:
        - None
        """
        for i in range(Tsim):
            # print("time: ", i)
            self.sample_environment(i)
            # print()

    def sample_environment(self, i: int) -> None:
        """
        Samples the environment using the sensors and sends data to the main CPU.

        This method samples the environment using the sensors and sends the collected
        data to the main CPU for processing.

        Parameters:
        - i (int): The current iteration in the simulation.

        Returns:
        - None
        """
        # ensure all data from buffer is propogated
        # print("outside bro: \n", self.broadcast_buffer)
        # print(f"\n\n t: {i}")
        # print("cpu broadcast times: \n", self.cpu_broadcast_times)
        self.send_data_to_cpu_from_buffer(t=i, buffer=self.buffer_nf, cpu=self.cpu)
        self.send_data_to_cpu_from_buffer(t=i, buffer=self.buffer_fb, cpu=self.feedback_cpu)
        self.broadcast_data_to_sensor_from_buffer(t=i, buffer=self.broadcast_buffer, cpu=self.feedback_cpu)

        # get the true state value
        new_x = self.step_environment(t=i)
        self.x['x'][i] = new_x
        # print("new_x: ", new_x)
        noise = np.random.normal(0, np.sqrt(self.omega_sq), size=len(new_x))
        x_noisy = new_x+noise
        # print("x_noisy: ", x_noisy)
        # pool the sensors and 
        self.pool_sensors(i=i, new_x=x_noisy, sensors=self.nf_sensors, buffer=self.buffer_nf)
        self.pool_sensors(i=i, new_x=x_noisy, sensors=self.fb_sensors, buffer=self.buffer_fb)
                

    def step_environment(self, t: int) -> np.ndarray:
        """
        Simulates environmental changes over time using a random walk.

        This method simulates the environmental changes over time using a random walk model.

        x[t+1] = x[t] + d
        choose d_i from uniform distribution with limits in self.d
        with probability self.p/2 choose from uniform[d_lower, d_upper] in self.d
        with probability self.p/2 choose from uniform[-d_upper, -d_lower] in self.d
        else: d = 0

        Parameters:
        - t (int): Current time in the simulation.

        Returns:
        - np.ndarray: New state vector representing the environmental changes.
        """
        if(t%self.del_T==0):
            d = np.zeros(self.n)
            for i in range(self.n):
                p = np.random.random()
                if p < self.p :
                    # do something
                    d[i] = np.random.uniform(self.d[0], self.d[1])
                    if p < self.p/2:
                        d[i] = -d[i]      
                else:
                    d[i] = 0

            if(t>0):
                last_x =  self.x['x'][t-1]
                return last_x + d
            else:
                return d
        else:
            return self.x['x'][t-1]

  
    def pool_sensors(self, i: int, new_x: np.ndarray, sensors: List[Sensor], buffer: pd.DataFrame) -> None:
        """
        Updates sensors with the new environment data and sends information to the CPU.

        This method updates all sensors with the new environment data, checks for triggers,
        and sends the information to the CPU for processing.

        Parameters:
        - i (int): Current iteration in the simulation.
        - new_x (np.ndarray): New environment state vector.
        - sensors (List[Sensor]): List of sensors to update.
        - buffer (pd.DataFrame): Data buffer for sensor-to-CPU communication.

        Returns:
        - None
        """
        for sensor in sensors:
            if(i-sensor.b>=0) and ((i-sensor.b)%sensor.tau ==0):
                # print("pool_sensors: ", i, sensor.id, new_x)
                triggered, id, idx_transmitting, y_hat =sensor.run(i=i, new_x=new_x)
                if(triggered):
                    new_data = pd.DataFrame({'type':sensor.type, 'id': id, 't_start': i, 't': i+len(idx_transmitting)*self.td[0], 'y_hat': [y_hat.copy()], 'idx_trans': [idx_transmitting.copy()]})
                    if(sensor.type=='nf'):
                        # integral is number of packets sent * dt
                        self.int_transm_nf[sensor.id]+=len(idx_transmitting)*(len(idx_transmitting)*self.td[0])
                        self.buffer_nf = pd.concat([self.buffer_nf, new_data], ignore_index=True, axis=0)
                        # print("buffer: \n", self.buffer_nf)
                    else:
                        self.int_transm_fb[sensor.id]+=len(idx_transmitting)*(len(idx_transmitting)*self.td[0])
                        self.buffer_fb = pd.concat([self.buffer_fb, new_data], ignore_index=True, axis=0)
                        # print("buffer: \n", self.buffer_fb)
    
    def send_data_to_cpu_from_buffer(self, t: int, buffer: pd.DataFrame, cpu: Central_processor) -> None:
        """
        Sends data from the buffer to the CPU for processing.

        This method processes the data stored in the buffer and sends it to the CPU for further computation.

        Parameters:
        - t (int): Current time in the simulation.
        - buffer (pd.DataFrame): Data buffer to be sent to the CPU.
        - cpu (Central_processor): CPU instance for processing the received data.

        Returns:
        - None
        """
        if not buffer.empty:
            # sort by time, so latest time is at the top
            data_to_send =  buffer[buffer['t']<=t].copy()
            # there is no data to send
            if data_to_send.empty:
                return
            
            # print("time: ", t, "data to send to cpu: \n", data_to_send)
            self.sensor_broadcast_times = pd.concat([self.sensor_broadcast_times, data_to_send], ignore_index=True)
            # remove these data from the buffer
            if cpu.type=='fb':
                self.buffer_fb =  buffer[buffer['t']>t]
            else:
                self.buffer_nf =  buffer[buffer['t']>t]
            # sort the data by time and send it sequentially to cpu. 
            data_to_send =  data_to_send.sort_values(by='t')
            # iterate over all t and send the data
            for index, data in data_to_send.iterrows():
                # print("dt", data.t)
                broadcast, x_hat, idx_to_broadcast, time_sampled = cpu.recieve_data(id=data['id'], t_start=data['t_start'], t=data['t'], y_hat=data['y_hat'], idx_trans = data['idx_trans'])
                if(broadcast):
                    # print("broadcast")
                    # print("broadcast")   
                    self.int_rec_fb+=len(idx_to_broadcast)*(len(idx_to_broadcast)*self.td[1])                                
                    new_data = pd.DataFrame({'type':cpu.type,         'time_sampled': [time_sampled.copy()], 
                                             't_start': data['t'],    't': data['t']+ len(idx_to_broadcast)*self.td[1], 
                                             'x_hat': [x_hat.copy()], 'idx_trans': [idx_to_broadcast.copy()]})
                    
                    self.broadcast_buffer = pd.concat([self.broadcast_buffer, new_data], ignore_index=True, axis=0)
                    # print("buffer:\n", self.broadcast_buffer)

    def broadcast_data_to_sensor_from_buffer(self, t: int, buffer: pd.DataFrame, cpu: Central_processor) -> None:
        """
        Broadcasts data from the CPU to the sensors.

        This method broadcasts data from the CPU to the sensors for further processing.

        Parameters:
        - t (int): Current time in the simulation.
        - buffer (pd.DataFrame): Data buffer to be broadcasted.
        - cpu (Central_processor): CPU instance for broadcasting to sensors.

        Returns:
        - None
        """
        if not buffer.empty:
            # print("broadcast")
            # sort by time, so latest time is at the top
            data_to_send =  buffer[buffer['t']<=t].copy()
            if(data_to_send.empty):
                return
            self.cpu_broadcast_times= pd.concat([self.cpu_broadcast_times, data_to_send], ignore_index=True, axis=0)
            # remove these data from the buffer
            self.broadcast_buffer =  buffer[buffer['t']>t]
            # sort the data by time and send it sequentially to sensors. 
            data_to_send =  data_to_send.sort_values(by='t')
            for _, data in data_to_send.iterrows():
                cpu.send_data_to_all_sensors(x_hat=data['x_hat'], time_sampled=data['time_sampled'], idx_trans = data['idx_trans'])

    def plot(self) -> None:
        """
        Generates and displays plots related to the simulation.

        This method generates and displays visual plots depicting communication and state trajectories.

        Returns:
        - None
        """        
        self.plot_comm()
        self.plot_states()

    def plot_comm(self) -> None:
        """
        Generates communication-related plots.

        This method generates plots visualizing communication between sensors and the central processor.

        Returns:
        - None
        """
        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,6))
        cmap = plt.get_cmap('Set1')

         # Set the limits of the axes
        ax1.set_xlim(left=0, right=self.Tsim+1)
        ax2.set_xlim(left=0, right=self.Tsim+1)

         # Add labels to the x-axes
        ax2.set_xlabel('Time (s)')
        ax1.set_ylabel('NF')
        ax2.set_ylabel('FB')
        nf = self.sensor_broadcast_times[self.sensor_broadcast_times['type']=='nf']
        fb = self.sensor_broadcast_times[self.sensor_broadcast_times['type']=='fb']
        self.plot_sensors(ax=ax1, sensor_broadcast_times=nf, cmap=cmap)
        self.plot_sensors(ax=ax2, sensor_broadcast_times=fb, cmap=cmap)
        self.plot_cpu_fb(ax=ax2)
       
        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower center')
        plt.autoscale()
        plt.suptitle('Communications between Sensors and Central Processor vs. Time')
        plt.show()


    def plot_states(self) -> None:
        """
        Generates state-related plots.

        This method generates plots illustrating the true and estimated state trajectories.

        Returns:
        - None
        """
        fig, ax = plt.subplots(self.n, 1, sharex=True, figsize=(20,3*self.n)) 
        ax[self.n-1].set_xlabel('Time (s)')
        
            
        nf = self.extend_df(self.cpu.x)
        fb = self.extend_df(self.feedback_cpu.x)
        true_state = self.x

        nf_x = np.array(nf['x'].values.tolist())
        nf_t = nf['t'].values.tolist()
        fb_x = np.array(fb['x'].values.tolist())
        fb_t = fb['t'].values.tolist()
        true_state_x = np.array(true_state['x'].values.tolist())
        true_state_t = self.x.index.to_list()

        # nf_t, nf_x = self.extend_line(t=nf_t, x=nf_x, T=self.Tsim)
        # fb_t, fb_x = self.extend_line(t=fb_t, x=fb_x, T=self.Tsim)
        for i in range(self.n):
            ax[i].set_xlim(left=0, right=self.Tsim+1)
            ax[i].plot(true_state_t, true_state_x[:,i], linestyle="-", label='True')
            ax[i].plot(nf_t, nf_x[:,i].tolist(), linestyle="--", label = 'NF')
            ax[i].plot(fb_t, fb_x[:,i].tolist(), linestyle="dotted", label='FB')
            ax[i].set_ylabel(f"x({i})")

        plt.legend(bbox_to_anchor=(1.05, 0), loc='lower center')
        plt.autoscale()
        plt.tight_layout()
        plt.suptitle('True and Estimated Trajectories vs. Time')
        plt.show()
            
    def extend_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extends the DataFrame to fill missing time steps.

        This method extends the DataFrame to fill missing time steps in the simulation.

        Parameters:
        - df (pd.DataFrame): DataFrame to be extended.

        Returns:
        - pd.DataFrame: Extended DataFrame with complete time steps.
        """
        # Convert 't' column to float
        df['t'] = df['t'].astype(float)

        # Create a new DataFrame with complete 't' values
        new_t_values = pd.DataFrame({'t': np.arange(df['t'].min(), self.Tsim)})

        # Merge to fill missing 't' values and corresponding 'x' values
        result = pd.merge(new_t_values, df, on='t', how='left')

        # Fill NaN values in 'x' column with previous values
        result['x'] = result['x'].fillna(method='ffill')
        return result


    def plot_sensors(self, ax, sensor_broadcast_times, cmap) -> None:
        """
        Generates sensor-related plots.

        This method generates plots visualizing sensor communication with the CPU.

        Parameters:
        - ax: Axes object for plotting.
        - sensor_broadcast_times (pd.DataFrame): DataFrame containing sensor broadcast times.
        - cmap: Color map for visualization.

        Returns:
        - None
        """
        sensor_id = sensor_broadcast_times['id'].tolist()
        t_start = sensor_broadcast_times['t_start'].tolist()
        t_receive = sensor_broadcast_times['t'].tolist()

    
        # Draw the horizontal lines
        ax.axhline(0, color='black', linestyle='--', linewidth=2)
        ax.axhline(1, color='black',linestyle='--', linewidth=2)

       # Draw the vertical lines
        for i in range(len(sensor_id)):
            ax.axvline(t_start[i], color='grey', alpha=0.4, linewidth=1)
            ax.axvline(t_receive[i], color='grey', alpha=0.4, linewidth=1)
            label = "Sensor "+ str(sensor_id[i]+1)
            color = cmap(sensor_id[i])
            if label not in ax.get_legend_handles_labels()[1]:  # Check if label doesn't exist in legend
                ax.plot([t_start[i], t_receive[i]], [0, 1], linestyle='-', linewidth=1.5, label=label, color=color)
            else:
                ax.plot([t_start[i], t_receive[i]], [0, 1], linestyle='-', linewidth=1.5, color=color)


        # Remove the ticks and labels from the y-axes
        ax.set_yticks([])
        ax.annotate('Sensor', (0, 0), ha='right', va='center', xytext=(-1.3, 0))
        ax.annotate('CPU', (0, 1), ha='right', va='center',  xytext=(-1.3, 1))
    
    def plot_cpu_fb(self, ax) -> None:
        """
        Generates CPU feedback-related plots.

        This method generates plots visualizing CPU feedback communication.

        Parameters:
        - ax: Axes object for plotting.

        Returns:
        - None
        """
        self.cpu_broadcast_times
        t_start = self.cpu_broadcast_times['t_start'].tolist()
        t_receive = self.cpu_broadcast_times['t'].tolist()
         # Draw the vertical lines
        for i in range(len(t_start)):
            ax.axvline(t_receive[i], color='red', alpha=0.4, linewidth=1)
            ax.plot([t_start[i], t_receive[i]], [1, 0], linestyle='-', linewidth=1, color='black')
    
    def calculate_performance_metrics(self, print_metircs=False) -> Tuple[float, float, float, float]:
        """
        Calculates performance metrics.

        This method computes performance metrics such as MSE and power expenditure.

        Parameters:
        - print_metircs (bool, optional): Flag to print metrics. Defaults to False.

        Returns:
        - Tuple[float, float, float, float]: Calculated MSE and power expenditure for NF and FB cases.
        """
        nf = self.extend_df(self.cpu.x)
        fb = self.extend_df(self.feedback_cpu.x)
        true_state = self.x

        nf_x = np.array(nf['x'].values.tolist())
        fb_x = np.array(fb['x'].values.tolist())
        true_state_x = np.array(true_state['x'].values.tolist())

        mse_nf = self.calculate_MSE(x=nf_x, true_x=true_state_x)
        mse_fb = self.calculate_MSE(x=fb_x, true_x=true_state_x)

        power_nf, power_fb = self.calculate_power_expenditure()
        # Print the values in a table-like format
        if(print_metircs):
            print("Metrics\t\tMSE\t\tPower Expenditure")
            print("NF:\t\t{:.4f}\t\t{}".format(mse_nf, power_nf))
            print("FB:\t\t{:.4f}\t\t{}".format(mse_fb, power_fb))

        return mse_nf, mse_fb, power_nf, power_fb

    def calculate_MSE(self, x, true_x) -> float:
        """
        Calculates Mean Squared Error (MSE) between estimated and true states.

        This method computes the Mean Squared Error (MSE) between estimated and true states.
        Approximate the integral using a runnning average. 
        
        Parameters:
        - x: Estimated states.
        - true_x: True states.

        Returns:
        - float: Calculated Mean Squared Error.
        """
        # dt = 1
        MSE_integral = np.square(true_x - x)
        return np.sum(MSE_integral)/self.Tsim
        
        

    def calculate_power_expenditure(self) -> Tuple[float, float]:
        """
        Calculates power expenditure for NF and FB cases.

        This method computes the power expenditure for NF and FB communication scenarios.

        Returns:
        - Tuple[float, float]: Calculated power expenditure for NF and FB cases.
        """
        nf = self.Pu*np.sum(self.int_transm_nf)
        fb = self.Pu*np.sum(self.int_transm_fb) + self.Pd*self.int_rec_fb
        return nf/self.Tsim,fb/self.Tsim
