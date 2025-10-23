import numpy as np
from typing import Dict, Any

def fftfreq(n, d=1.0):
    """
    A variant on numpy.fft.fftfreq which flips the "negative" frequencies to
    frequencies above the Nyquist frequency.  When this is used,
    numpy.fft.fftshift() should NOT be used.
    """
    return np.linspace(0, n, n, endpoint=False) / (n*d)

def fftshift(x):
    """
    A variant on numpy.fft.fftshift that does NOT actually shift.  It is
    defined to be consistent with fftfreq() above.
    """
    return x



class DataAttr:
    """
    Wrapper class to associate metadata attributes (attr) with a numpy array (data).
    """
    def __init__(self, data: np.ndarray, attr: Dict[str, Any]):
        self.data = data
        self.attr = attr

    def __getitem__(self, key):
        """
        Attempts to retrieve metadata attribute using dictionary key access.
        """
        return self.attr[key]

