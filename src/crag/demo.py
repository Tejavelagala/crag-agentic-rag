"""Compatibility alias for the demo service module."""

import sys

from crag.services import demo as _demo

sys.modules[__name__] = _demo
