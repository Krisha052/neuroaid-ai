class ScreeningError(Exception):
    """Base class for expected, user-facing screening errors."""
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class InvalidInputError(ScreeningError):
    """Malformed or missing request input (e.g. no file, wrong field)."""
    status_code = 400

class UnsupportedMediaTypeError(ScreeningError):
    """File was provided but isn't a format we accept."""
    status_code = 415

class PayloadTooLargeError(ScreeningError):
    """Upload exceeds configured size/duration limits."""
    status_code = 413

class UnprocessableAudioError(ScreeningError):
    """File was accepted but couldn't be read/decoded as audio."""
    status_code = 422
