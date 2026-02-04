class TNPIError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class ConfirmError(TNPIError):
    pass
