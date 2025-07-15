import typing as T
import numpy as np
import gymnasium as gym
import time
import mujoco
from mujoco import MjModel, MjData
import os
from scipy.spatial.transform import Rotation as R
from mujoco.viewer import launch_passive


class SelfBalancingRobotEnv(gym.Env):
    
    def __init__(self, environment_path: str = "./models/scene.xml", max_time: float = 10.0, max_pitch: float = 0.8, frame_skip: int = 10):
        """
        Initialize the SelfBalancingRobot environment.
        
        Args:
            environment_path (str): Path to the MuJoCo model XML file.
        """
        # Initialize the environment
        super().__init__()
        self.viewer = None
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        self.max_time = max_time  # Maximum time for the episode
        self.frame_skip = frame_skip  # Number of frames to skip in each step   

        # Action and observation spaces
        # Observation space: pitch, roll, yaw, body_ang_vel_x, body_ang_vel_y, linear_vel_x, linear_vel_y, pos_x, pos_y
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)

        # Action space: torque applied to the wheels
        self.action_limit = 10.0
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float64)

        self.weight_fall_penalty = 100.0  # Penalty for falling
        self.max_pitch = max_pitch  # Maximum pitch angle before truncation
        self.count_pos = 0
        self.last_position = np.zeros(2)  # Last position of the robot
        self.count_yaw = 0
        self.last_yaw = 0.0  # Last yaw angle
        self.last_direction = np.zeros(2)
        self.count_dir = 0
        
        

    def step(self, action: T.Tuple[float, float]) -> T.Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step in the environment.
        
        Args:
            action (np.ndarray or list): The action to be taken, which is a torque applied
            to the wheels of the robot.

        Returns:
            T.Tuple[np.ndarray, float, bool, dict]: A tuple containing:
                - obs (np.ndarray): The observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
                - reward (float): The reward received after taking the action   
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode has been truncated.
                - info (dict): Additional information about the environment.
        """
        self.data.ctrl[:] = np.clip(action, -self.action_limit, self.action_limit)  # Apply action (torque to the wheels)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # Step the simulation
        obs = self._get_obs()
        reward = self._compute_reward(np.array(action)) # Compute the reward based on the action taken
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Penalità di caduta al termine dell'episodio
        x, y, z = self._get_position()  # Ottieni la posizione del robot
        position = np.array([x, y])
        pos_displacement = np.linalg.norm(position - self.last_position)
        roll, pitch, yaw = self._get_body_orientation_angles()
        yaw_displacement = abs(yaw - self.last_yaw)
        linear_vel_x, linear_vel_y, linear_vel_z = self._get_robot_linear_velocity() # Velocità lineare del robot (x, y, z)
        linear_norm = np.linalg.norm([linear_vel_x, linear_vel_y])
        if truncated:
            reward -= (self.weight_fall_penalty + 10 * yaw_displacement + 10 * pos_displacement)
        elif terminated and (pos_displacement < 0.1):
            reward += 400  # Bonus for staying still at the end of the episode

        if terminated and (abs(pitch) < 0.08):
            reward += 100 - np.dot([linear_vel_x, linear_vel_y], self.last_direction)
        
        if self.count_dir == 5:
            self.count_dir = 0
            self.last_direction = [linear_vel_x, linear_vel_y]
        self.count_dir += 1


        # info = self._get_info()
        return obs, reward, terminated, truncated, {}
        
    def reset(self, seed: T.Optional[int] = None, options: T.Optional[dict] = None) -> np.ndarray:
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for resetting the environment.
            
        Returns:
            T.Dict[str, T.Any]: A dictionary containing the initial observation of the environment (pitch angle, pitch velocity, x position, and x velocity).
        """
        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # Reset the simulation data
        self._initialize_random_state()
        self._last_action = np.zeros(2) # Resetta anche l'ultima azione
        # info = self._get_info()
        obs = self._get_obs()
        return obs, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode in which to render the environment. Default is 'human'.
        """
        if self.viewer is None:
            self.viewer = launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.model.opt.timestep * self.frame_skip)  # Sleep for the duration of the frame skip
        else:
            raise RuntimeError("Viewer is not running. Please reset the environment or start the viewer.")

    # --- Nuove funzioni di supporto per la reward ---

    def _get_body_orientation_angles(self) -> T.Tuple[float, float, float]:
        """
        Estrae gli angoli di roll, pitch, yaw dal corpo principale del robot.
        Questo dipende da come il tuo robot è modellato in MuJoCo.
        Normalmente il quaternione del root body si trova in data.qpos[3:7].
        """
        # Converti il quaternione MuJoCo [w, x, y, z] a scipy [x, y, z, w]
        quat_wxyz = self.data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        euler = R.from_quat(quat_xyzw).as_euler('xyz')
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        return float(roll), float(pitch), float(yaw)

    def _get_body_angular_velocities(self) -> np.ndarray:
        """
        Estrae le velocità angolari del corpo principale del robot.
        Queste sono spesso disponibili in data.qvel per il root body (indices 3:6).
        """
        # Assumiamo che le velocità angolari del corpo libero siano in data.qvel[3:6]
        return self.data.qvel[3:6]

    def _get_robot_linear_velocity(self) -> np.ndarray:
        """
        Ottiene la velocità lineare del robot (spesso del centro di massa o di un corpo specifico).
        Assumiamo che le velocità lineari del corpo libero siano in data.qvel[0:3]
        """
        return self.data.qvel[0:3]
    
    def _get_position(self) -> np.ndarray:
        """
        Ottiene la posizione del robot nel mondo.
        Assumiamo che la posizione del corpo principale sia in data.qpos[0:3].
        """
        return self.data.qpos[0:3]

    def _kernel(self, x: float, alpha: float) -> float:
        """
        Gaussian kernel function for reward computation.
        Args:
            x (float): The input value.
            alpha (float): The bandwidth parameter for the Gaussian kernel. 
        Returns:
            float: The value of the Gaussian kernel at x.
        """
        return np.exp(-(x**2)/alpha)

    # --- Funzione _compute_reward focalizzata sul bilanciamento e la stasi ---
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute the reward for the current step, focused on self-balancing and staying still.

        Args:
            action (np.ndarray): The action taken in the environment (es. comandi motori).

        Returns:
            float: The computed reward.
        """
        roll, pitch, yaw = self._get_body_orientation_angles() # Ottieni roll, pitch, yaw
        yaw_displacement = abs(yaw - self.last_yaw)  # Calcola lo spostamento angolare rispetto all'ultimo yaw
        yaw_displacement_penalty = self._kernel(yaw_displacement, alpha=0.01)  # Penalità per spostamento angolare
        if self.count_yaw == 10:
            self.last_yaw = yaw
            self.count_yaw = 0
        self.count_yaw += 1

        if abs(yaw) < 0.08:
            self.last_position = self.data.qpos[:2].copy()  # Aggiorna l'ultima posizione solo se il robot è in stasi rotazionale

        x, y, z = self._get_position()  # Ottieni la posizione del robot
        position = np.array([x, y])
        pos_displacement = np.linalg.norm(position - self.last_position)  # Calcola lo spostamento rispetto all'ultima posizione
        pos_displacement_penalty = self._kernel(pos_displacement, alpha=0.1)  # Penalità per lo spostamento
        # if self.count_pos == 5:
        #     self.last_position = position
        #     self.count_pos = 0
        # self.count_pos += 1

        linear_vel_x, linear_vel_y, linear_vel_z = self._get_robot_linear_velocity() # Velocità lineare del robot (x, y, z)
        linear_norm = np.linalg.norm([linear_vel_x, linear_vel_y])
        linear_penalty = self._kernel(linear_norm, alpha=0.0001) # Penalità per velocità lineare

        # Dati motori/azioni
        torques = self.data.ctrl # Torque effettivamente applicati o comandi motore
        torque_norm = np.linalg.norm(torques)
        torque_penalty = self._kernel(torque_norm, alpha=0.5)

        if pos_displacement == 0.0:
            reward = yaw_displacement_penalty * torque_penalty * linear_penalty
        else:
            reward = yaw_displacement_penalty * torque_penalty * pos_displacement_penalty

        return reward

    # --- Le seguenti funzioni non sono più necessarie o sono state inglobate ---
    # Le ho lasciate commentate per chiarezza, non rimuoverle se vuoi mantenere il file pulito ma con storia.
    # def _pitch_reward_component(self, alpha: float) -> float:
    #     pass 

    # def _velocity_reward_component(self, alpha: float) -> float:
    #     pass 

    # def _wheels_reward_component(self, alpha: float) -> T.Tuple[float, float, float]:
    #     pass 

    # def _gyro_reward_component(self, alpha: float) -> float:
    #     pass 

        
    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: The observation vector.
            
            Ho mantenuto l'output di 7 elementi, che ora include:
            [pitch, roll, yaw, body_ang_vel_x, body_ang_vel_y, linear_vel_x, linear_vel_y]
        """
        roll, pitch, yaw = self._get_body_orientation_angles()
        body_ang_vel = self._get_body_angular_velocities() # [gyro_x, gyro_y, gyro_z]
        linear_vel = self._get_robot_linear_velocity() # [vel_x, vel_y, vel_z]
        x, y, z = self._get_position() # [pos_x, pos_y, pos_z]
        
        return np.array([
            pitch,          # Inclinazione avanti/indietro (fondamentale per bilanciamento)
            roll,           # Inclinazione laterale (fondamentale per bilanciamento)
            yaw,            # Orientamento sul piano orizzontale (per la stasi rotazionale)
            body_ang_vel[1], # Velocità angolare asse Y del corpo (pitch rate)
            body_ang_vel[2], # Velocità angolare asse Z del corpo (yaw rate)
            linear_vel[0],  # Velocità lineare in avanti (asse X) (per la stasi lineare)
            linear_vel[1],  # Velocità lineare laterale (asse Y) (per la stasi lineare)
            x,              # Posizione del robot (asse X)
            y,              # Posizione del robot (asse Y)
        ], dtype=np.float64)


    def _get_info(self):
        # Questo metodo non è stato modificato in quanto non è stato specificato.
        # Potrebbe essere utile per debug o per raccogliere metriche.
        return {}

    def _is_terminated(self) -> bool:
        """
        Check if the episode is terminated.
        
        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        # Terminated quando il robot cade o raggiunge la fine del tempo massimo
        return self._is_truncated() or self.data.time >= self.max_time
    
    def _is_truncated(self) -> bool:
        # Ottieni orientamento attuale in angoli Euler
        roll, pitch, yaw = self._get_body_orientation_angles()

        # Truncate if the pitch or roll angle is too high (robot falls)
        return bool(abs(pitch) > self.max_pitch or abs(roll) > 0.06 ) # pitch, roll, yaw thresholds


    def _initialize_random_state(self):
        # Reset position and velocity
        self.data.qpos[:3] = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.25]  # Initial position (x, y, z)
        self.last_position = self.data.qpos[:2].copy()
        self.data.qvel[:] = 0.0  # Initial speed

        # Euler angles: Roll=0, Pitch=random, Yaw=random
        euler = [
            0.0, # Roll
            np.random.uniform(-0.6, 0.6), # Pitch
            np.random.uniform(-np.pi, np.pi) # Yaw
        ]

        # Euler → Quaternion [x, y, z, w]
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        self.data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        self.last_yaw = euler[2]