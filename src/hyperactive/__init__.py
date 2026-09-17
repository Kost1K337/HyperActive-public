"""HyperActive: reinforcement learning for well development planning.

Package layout:

* :mod:`hyperactive.core` - wells, tasks, crews, plans;
* :mod:`hyperactive.planning` - crew scheduling, production profiles, NPV, constraints;
* :mod:`hyperactive.greedy` - greedy baseline planner;
* :mod:`hyperactive.env` - Gymnasium environment and candidate features;
* :mod:`hyperactive.models` - masked C-DQN with behavioural cloning;
* :mod:`hyperactive.inference` - loading a trained policy and planning with it;
* :mod:`hyperactive.data` - input loaders and the synthetic data generator.
"""

__version__ = "0.1.0"
