"""Global defaults for original MO-CAPO datasets.

Fairness datasets override development, few-shot, test, and block sizes through
``DatasetConfig.fairness``.
"""

from tri_fair.config.schema import SetupConfig

SETUP = SetupConfig(dev_size=300, fs_size=100, test_size=500, n_steps=2000)
SETUP.validate()
