"""Compatibility alias for the provider composition service."""

import sys

from crag.services import composition as _composition

sys.modules[__name__] = _composition
