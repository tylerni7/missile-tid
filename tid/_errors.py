"""
Errors that may be thrown and caught specific to TID usage
"""


class TidRuntimeError(RuntimeError):
    """
    generic TID error that is likely unrecoverable and requires user intervention
    """
