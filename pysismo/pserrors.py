"""
Module managing errors
"""


class CannotPreprocess(Exception):
    """
    Cannot preprocess station waveform
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class NaNError(Exception):
    """
    Got NaN in numpy array
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class NoPAZFound(Exception):
    """
    Could not find PAZ
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class CannotCalculateInstFreq(Exception):
    """
    Cannot calculate instantaneous frequency in FTAN
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class CannotPerformTomoInversion(Exception):
    """
    Cannot perform tomographic inversion (e.g., because no data is available)
    """
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)