from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.distributions import Distribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy  
from stable_baselines3.common.utils import zip_strict
from typing import Tuple


class GruActorCriticPolicy(ActorCriticPolicy):
    """
    Only think of sharing GRU W/ share_features_extractor
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gru_hidden_size: int = 256,
        n_gru_layers: int = 1,
        enable_critic_gru: bool = True,
        gru_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.gru_output_size = gru_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        assert  (
            self.share_features_extractor
        ), "You must choose between shared Extractor."
        
        self.gru_kwargs = gru_kwargs or {}
        self.enable_critic_gru = enable_critic_gru
        
        self.gru = nn.GRU(self.features_dim, 
                        gru_hidden_size,
                        num_layers = n_gru_layers,
                        **self.gru_kwargs,)
        
        self.gru_hidden_state_shape = (n_gru_layers, 1, gru_hidden_size)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.gru_output_size,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        hidden_state: th.Tensor,
        episode_starts: th.Tensor,
        gru: nn.GRU,
    ) -> Tuple[th.Tensor, th.Tensor]:
       
        n_seq = hidden_state.shape[1]
        features_sequence = features.reshape((n_seq, -1, gru.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if th.all(episode_starts == 0.0):
            gru_output, hidden_state = gru(features_sequence, hidden_state)
            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)
            return gru_output, hidden_state
        
        gru_output = []
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            gru_hidden, hidden_state = gru(features.unsqueeze(dim=0), (1.0 - episode_start).view(1, n_seq, 1) * hidden_state)
            gru_output += [gru_hidden]
        # Sequence to batch
        gru_output = th.flatten(th.cat(gru_output).transpose(0, 1), start_dim=0, end_dim=1)
        return gru_output, hidden_state

    def forward(
        self,
        obs: th.Tensor,
        gru_states: th.Tensor,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param gru_states: The last hidden and memory states for the GRU.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the gru states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        ### think of two options
        ### 1.  sharing gru w/ sharing extractor
        ### 2.  sharing gru w/o sharing extractor

        features = self.extract_features(obs) ## Extractor (e.g. CNN extractor in here)

        latent_pi, gru_states = self._process_sequence(features, gru_states, episode_starts, self.gru)
        copy_latent = latent_pi.detach()# Re-use LSTM features but do not backpropagate
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(copy_latent)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, gru_states

    def get_distribution(
        self,
        obs: th.Tensor,
        gru_states: th.Tensor,
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, th.Tensor]: 
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param gru_states: The last hidden and memory states for the GRU.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the gru states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent, gru_states = self._process_sequence(features, gru_states, episode_starts, self.gru)
        latent_pi = self.mlp_extractor.forward_actor(latent)
        return self._get_action_dist_from_latent(latent_pi), gru_states

    def predict_values(
        self,
        obs: th.Tensor,
        gru_states: th.Tensor,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param gru_states: The last hidden and memory states for the GRU.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the gru states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        latent_pi, _ = self._process_sequence(features, gru_states, episode_starts, self.gru)
        latent_vf = latent_pi.detach()
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, gru_states: th.Tensor, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param gru_states: The last hidden and memory states for the GRU.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the gru states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, gru_states = self._process_sequence(features, gru_states, episode_starts, self.gru)
        latent_vf = latent_pi.detach()
        
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        gru_states: th.Tensor,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param gru_states: The last hidden and memory states for the GRU.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the gru states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, gru_states = self.get_distribution(observation, gru_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), gru_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)
        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.gru_hidden_state_shape) for _ in range(n_envs)], axis=1)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state, dtype=th.float32, device=self.device)

            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, gru_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = states.cpu().numpy()

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

class GruActorCriticCnnPolicy(GruActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gru_hidden_size: int = 256,
        n_gru_layers: int = 1,
        enable_critic_gru: bool = True,
        gru_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            gru_hidden_size,
            n_gru_layers,
            enable_critic_gru,
            gru_kwargs
        )


class GruMultiInputActorCriticPolicy(GruActorCriticPolicy):
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        gru_hidden_size: int = 256,
        n_gru_layers: int = 1,
        enable_critic_gru: bool = True,
        gru_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            gru_hidden_size,
            n_gru_layers,
            enable_critic_gru,
            gru_kwargs
        )