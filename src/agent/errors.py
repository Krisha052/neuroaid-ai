from src.api.errors import ScreeningError


class AgentError(ScreeningError):
    """Base class for agent-orchestration failures."""
    status_code = 502

class AgentNotConfiguredError(AgentError):
    """ANTHROPIC_API_KEY isn't set on this server."""
    status_code = 503

class MaxIterationsExceededError(AgentError):
    """Agent didn't reach a terminal tool call within the iteration budget."""
    status_code = 502

class AgentUpstreamError(AgentError):
    """The Anthropic API itself returned an error (auth, billing, rate limit,
    outage, ...) -- distinct from a bug in our own orchestration code."""
    status_code = 502
