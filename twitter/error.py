class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class TotalRateLimitError(Error):
    """Exception raised when all APIs get unavailable."""
    def __init__(self, min_break_secs):
        self.min_break_secs = min_break_secs
