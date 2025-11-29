import numpy as np

class ReplayBuffer:
    """
    The base class for a replay buffer: stores gcsl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class
    """

    def __init__(self,
                observation_shape,
                observation_dtype,
                action_dim,
                action_dtype,
                goal_dim,
                goal_dtype,
                max_trajectory_length,
                her_ratio,
                buffer_size,
            ):
        """
        Args:
            env: A gcsl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
        """        
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length-1, action_dim),
            dtype=action_dtype
        )
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *observation_shape),
            dtype=observation_dtype
        )
        self._achieved_states = np.zeros(
            (buffer_size, max_trajectory_length, goal_dim),
            dtype=goal_dtype
        )
        self._desired_states = np.zeros(
            (buffer_size, goal_dim),
            dtype=goal_dtype
        )
                
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.her_ratio = her_ratio
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        
    def add_trajectory(self, states, actions, desired_state, achieved_states, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        assert actions.shape == self._actions[0].shape
        assert states.shape == self._states[0].shape

        self._actions[self.pointer] = actions
        self._states[self.pointer] = states
        self._desired_states[self.pointer] = desired_state
        self._achieved_states[self.pointer] = achieved_states
        if length_of_traj is None:
            length_of_traj = self.max_trajectory_length
        self._length_of_traj[self.pointer] = length_of_traj

        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size
    
    def _sample_indices(self, batch_size):
        pairs = np.column_stack(np.triu_indices(self.max_trajectory_length, k=1))
        np.random.shuffle(pairs)

        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)
        time_idxs = np.random.choice(pairs.shape[0], batch_size)
        pairs_batch = pairs[time_idxs]

        return traj_idxs, pairs_batch[:, 0], pairs_batch[:, 1]

    def sample_goals(self, batch_size):
        """
        Samples a batch of goals
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            goals
        """

        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)
        return self._states[traj_idxs, -1], self._achieved_states[traj_idxs, -1]
    
    def sample_initial_states(self, batch_size):
        """
        Samples a batch of goals
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            goals
        """

        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)
        return self._states[traj_idxs, 0], self._achieved_states[traj_idxs, 0]

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def get_transition(self, traj_idx, time_idx):
        assert time_idx.all() > 0
        next_state = self._states[traj_idx, time_idx]
        action = self._actions[traj_idx, time_idx-1]
        state = self._states[traj_idx, time_idx-1]
        achieved_state = self._achieved_states[traj_idx, time_idx-1]
        return state, action, next_state, achieved_state
    
    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self._states[traj_idxs, time_state_idxs]
        actions = self._actions[traj_idxs, time_state_idxs]
        achieved_states = self._achieved_states[traj_idxs, time_state_idxs]
        goals = self._desired_states[traj_idxs]
        lengths = self.max_trajectory_length - time_state_idxs
        horizons = np.expand_dims(self.max_trajectory_length - time_state_idxs, axis=1)

        if self.her_ratio > 0:
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.her_ratio)
            goals[her_indexes] = self._achieved_states[traj_idxs[her_indexes], time_goal_idxs[her_indexes]]        
            lengths[her_indexes] = time_goal_idxs[her_indexes] - time_state_idxs[her_indexes]
            horizons[her_indexes, :] = np.expand_dims(time_goal_idxs[her_indexes] - time_state_idxs[her_indexes], axis=1)

        weights = np.ones(batch_size)

        return observations, actions, goals, lengths, horizons, weights, achieved_states
    
    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
            achieved_states=self._achieved_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name)
        states, actions, desired_states, achieved_states = data['states'], \
            data['actions'], data['desired_states'], data['achieved_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], \
                                desired_states[i], achieved_states[i])

