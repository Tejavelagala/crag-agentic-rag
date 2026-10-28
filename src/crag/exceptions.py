"""Application-specific exception hierarchy."""


class CragError(Exception):
    """Base class for expected CRAG application errors."""


class ConfigurationError(CragError):
    """Raised when application configuration is invalid."""


class IntegrationError(CragError):
    """Raised when a configured external integration fails."""
