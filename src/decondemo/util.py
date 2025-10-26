import numpy as np
from typing import Dict, Any, Iterable, List, TypeVar

T = TypeVar('T')

def linear_size(a, b):
    if isinstance(a, np.ndarray):
        a = a.shape[0]
    if isinstance(b, np.ndarray):
        b = b.shape[0]
    return a+b-1

def zero_pad(array: np.ndarray, size: int) -> np.ndarray:
    """
    Pads an array with zeros up to the specified size.
    """
    if size < len(array):
        raise ValueError("Target size must be greater than or equal to array length.")
    
    padded_array = np.zeros(size, dtype=array.dtype)
    padded_array[:len(array)] = array
    return padded_array


def nhalf(N):
    '''
    Return the number of "positive" frequency samples in a sampling of size N.

    This excludes zero frequency and any Nyquist frequency sample.  As well as
    any frequency samples above the Nyquist frequency (aka "negative
    frequencies").
    '''
    if N%2: return (N-1)//2
    return (N-2)//2

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

def tee_and_capture(source_generator: Iterable[T], capture_list: List[T], name="") -> Iterable[T]:
    """
    Wraps a generator, yielding its items while simultaneously appending them
    to a provided list.
    """
    for item in source_generator:
        print(f'tee {name} {item.shape} tot:{np.sum(item)}')
        capture_list.append(item.copy())
        yield item


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
