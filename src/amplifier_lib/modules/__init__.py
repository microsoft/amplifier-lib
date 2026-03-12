"""Module resolution and activation for amplifier-lib.

This module provides basic module resolution - downloading modules from URIs
and making them importable. This enables a turn-key experience where bundles
can be loaded and executed without additional libraries.

For advanced resolution strategies (layered resolution, settings-based overrides,
workspace conventions), see amplifier-module-resolution.
"""

from amplifier_lib.modules.activator import ModuleActivationError, ModuleActivator

__all__ = ["ModuleActivator", "ModuleActivationError"]
