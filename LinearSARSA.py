import numpy as np
from tqdm import tqdm

class LinearSARSA:
    def __init__(
        self,
        env,
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        state_normailize = [4.8, 4, 0.4, 4],    # Only for CartPole evironment 
    ) -> None:
        """
        Initialize the LinearSARSA agent.
        
        Parameters
        ----------
        env : gym.Env
            The reinforcement learning environment (gymnasium assumed).
        learning_rate : float, optional
            Learning rate (alpha) for weight updates.
        discount_factor : float, optional
            Discount factor (gamma) for future rewards.
        epsilon_start : float, optional
            Initial exploration rate for epsilon-greedy policy.
        epsilon_min : float, optional
            Minimum exploration rate
        epsilon_decay : float, optional
            Decay rate for epsilon after each episode.
        state_normailize : list, optional
            Normalization factors for state variables, 
            by default [4.8, 4, 0.4, 4] (specific to CartPole)
        """
        
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.discount_factor = discount_factor
        self.lrate = learning_rate
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_normailize = np.array(state_normailize)
        self.weights = None
        
    def _get_action(self, q_values, train_mode=True):
        """
        Select an action using epsilon-greedy policy.
        
        Parameters
        ----------
        q_values : np.ndarray
            Q-values for all actions in the current state.
        train_mode : bool, optional
            Whether to use exploration (epsilon-greedy) during training, 
            by default True. If False, always selects greedy action.
        
        Returns
        -------
        int
            Selected action index.
        """
        
        if train_mode and (np.random.rand() < self.epsilon):
            return self.env.action_space.sample()
        
        return np.argmax(q_values)

    def _update_epsilon(self):
        """
        Decay the exploration rate (epsilon) exponentially.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, n_episodes=1000):
        """
        Train the LinearSARSA agent for a specified number of episodes.
        
        Parameters
        ----------
        n_episodes : int, optional
            Number of training episodes.
            
        Returns
        -------
        list
            List of total rewards obtained in each episode.
        """
        
        rewards = []
        self.weights = np.zeros((self.state_size, self.action_size))
        
        for _ in tqdm(range(n_episodes)):
   
            state, _ = self.env.reset()
            state /= self.state_normailize
            sum_reward = 0  
            
            q_values = state @ self.weights
            action = self._get_action(q_values)
    
            while True:
                
                state_next, reward, done, truncated, _ = self.env.step(action)
                state_next /= self.state_normailize
                sum_reward += reward
                
                q_values_next = state_next @ self.weights
                action_next = self._get_action(q_values_next)
                
                td_error = reward + self.discount_factor*q_values_next[action_next]-q_values[action]
                self.weights[:,action] += self.lrate * td_error * state
                
                if done or truncated:
                    break
                
                state = state_next  
                action = action_next
                q_values = q_values_next
                
            rewards.append(sum_reward)
            self._update_epsilon()

        return rewards
    
    def test(self, env, n_episodes=10) -> float:
        """
        Test the trained agent on the given environment.
        
        Parameters
        ----------
        env : gym.Env
            The environment to test the agent on.
        n_episodes : int, optional
            Number of test episodes.
        
        Returns
        -------
        list
            List of total rewards obtained in each test episode.
        
        Raises
        ------
        ValueError
            If the agent has not been trained (weights are None).
        
        Notes
        -----
        During testing, the agent uses greedy policy (no exploration).
        Renders the environment during testing for visualization.
        """
        if self.weights is None:
            raise ValueError("train method is not called")
        
        rewards = []
        
        for _ in range(n_episodes):
            state, _= env.reset()
            sum_reward = 0
            while True:
                q_values = state @ self.weights
                action = self._get_action(q_values, False)
                
                state_next, reward, done, truncated, _ = env.step(action)
                sum_reward += reward           
                
                if done or truncated:
                    break
                    
                state = state_next 
                env.render() 
                    
            rewards.append(sum_reward)
        env.close()

        return rewards