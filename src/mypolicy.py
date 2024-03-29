from typing import Any, Dict, List, Optional, Type, Union, Callable

import gym
import torch
from stable_baselines3.common.policies import (
    BasePolicy,
    ContinuousCritic,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch
)
# from stable_baselines3.common.type_aliases import Schedule
from torch import nn
Schedule = Callable[[float], float]


class SockfarmActor(BasePolicy):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        subnets: int = 2,
        user_dim: int = 10,
        max_requests: int = 100,
        max_prod: int = 10,
    ):
        super(SockfarmActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        self.subnets = subnets
        self.user_dim = user_dim
        self.max_requests = max_requests
        self.max_prod = max_prod

        action_dim = get_action_dim(self.action_space)
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        sub_action_dim = action_dim // 2

        partition_net = create_mlp(features_dim, self.user_dim, net_arch, activation_fn, squash_output=True)
        actor_net1 = create_mlp(features_dim, sub_action_dim, net_arch, activation_fn, squash_output=True)
        actor_net2 = create_mlp(features_dim, action_dim - sub_action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        self.part = nn.Sequential(*partition_net)
        self.mu1 = nn.Sequential(*actor_net1)
        self.mu2 = nn.Sequential(*actor_net2)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        # print(f"feature_shape: {features.shape}, obs_shape: {obs.shape}")
        # print(f"user_dim: {self.user_dim}, max_req: {self.max_requests}, max_prod: {self.max_prod}")

        ret1 = self.mu1(features)
        ret2 = self.mu2(features)

        if self.subnets > 1:
            part = self.part(features)
            inds = torch.argsort(part)
            ret = torch.cat([ret1, ret2]).view((self.user_dim, self.max_requests))[inds].view(1, -1)
        else:
            ret = torch.cat([ret1, ret2]).view(1, -1)

        # print(f"dim: {ret1.shape} {ret2.shape} -> {ret.shape}")
        # print(ret.shape)
        return ret

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(observation, deterministic=deterministic)


class SockfarmPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        subnets: int = 2,
        user_dim: int = 10,
        max_requests: int = 100,
        max_prod: int = 10,
    ):
        super(SockfarmPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [400, 300]
            else:
                net_arch = []

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.subnets = subnets
        self.user_dim = user_dim
        self.max_requests = max_requests
        self.max_prod = max_prod

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SockfarmActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SockfarmActor(**actor_kwargs,
                             subnets=self.subnets,
                             user_dim=self.user_dim,
                             max_requests=self.max_requests,
                             max_prod=self.max_prod,
                             ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic=deterministic)
