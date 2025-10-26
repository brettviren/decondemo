#!/usr/bin/env python
'''
This module implements a "chunked streaming" graph that performs signal
processing and other transformations on long arrays by breaking those arrays
into smaller chunks.

Some nodes of the graph may perform convolution or deconvolution.  These
operations internally require changing the size of chunks and require buffering
intermediate results for application to subsequent chunks.

For example, convolution produces a result extended by the size of the kernel
(minus one) by appending a padded region.  This padded region must be clipped,
retained and finally added to the beginning of the subsequent result.

Conversely, deconvolution requires a prepended padding.  This must be added to
the end of the previous result.  In order to achieve this, a deconvolution node
must buffer the prior result and emit it only after it receives and processes a
subsequent input.

All nodes are instances of a callable class.  The call method will yield
results.  The graph is composed through calls to the callable method.

'''
import random
import numpy as np

from .util import zero_pad
from .convo import convo as convo_func

class ExpoTime:
    def __init__(self, rate=1.0):
        self.rate = rate
    def __call__(self):
        t = random.expovariate(self.rate)
        return t

class UniformTime:
    def __init__(self, rate=1.0):
        self.rate = rate
    def __call__(self):
        return 1.0/self.rate

class ArrayTimeSource:
    def __init__(self, times):
        self.times = iter(times)
    
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            return next(self.times)
        except StopIteration:
            raise StopIteration

class TimeSource:
    '''
    A generator of decay times.
    '''
    def __init__(self, step, start=0.0, limit=100):
        '''
        Create a time source starting at "start" producing times via a
        "step" callable.  Each call to step is expected to return a relative
        time value that will advance the time source's time.

        Exit after emitting "limit" number of times.
        '''
        self.step = step
        self.now = start
        self.limit = limit

    def __call__(self):
        while self.now < self.limit:
            dt = self.step()
            self.now += dt
            yield self.now

class Latch:
    '''
    Produce a discrete sampled impulse train from times
    '''
    def __init__(self, sample_period=1.0, chunk_size=100, start_time=0.0):
        '''
        Latch input times with given sample period and emit arrays of chunk size.
        '''
        print(f'Latch: {sample_period=} {chunk_size=} {start_time=}')
        self.sample_period = sample_period
        self.chunk_size = chunk_size
        self.now = start_time
        self.chunk = None
        self.fresh()            # ignore old on first call

    def fresh(self):
        '''
        Initialize a fresh chunk, return the old chunk.
        '''
        old = self.chunk
        self.chunk = np.zeros(self.chunk_size, dtype=float)
        return old

    @property
    def duration(self):
        '''
        The duration of a chunk.
        '''
        return self.sample_period * self.chunk_size

    @property
    def later(self):
        '''
        The smallest past this chunk.
        '''
        return self.now + self.duration
        
    def contains(self, time):
        '''
        Return true if time is in the current chunk.
        '''
        return self.now <= time and time < self.now + self.duration

    def tick(self, time):
        '''
        Return the index for the time given current now time.

        This may be outside the bounds of the current chunk.  It will be
        negative if "time" is before "now".
        '''
        # Use np.floor to ensure correct behavior for negative indices (times before self.now)
        return int( np.floor( (time - self.now ) / self.sample_period ) )

    def latch(self, time, value=1.0):
        '''
        Add value the array at tick corresponding to time.

        Raise IndexError if time is out-of-bounds.
        '''
        ind = self.tick(time)
        if ind < 0 or ind >= self.chunk_size:
            raise IndexError(f'ind {ind} out of bounds for size {self.chunk_size}')
        self.chunk[ind] += value

    def emit(self):
        '''
        Yield current chunk and advance one duration.
        '''
        done = self.fresh()
        self.now += self.duration
        yield done

    def __call__(self, time_source):
        '''
        Emit chunks of latched time
        '''
        for time in time_source:
            if time < self.now:
                raise ValueError(f'time violation {time} < {self.now}')

            # Emit chunks until we catch up
            while time >= self.later:
                for chunk in self.emit():
                    yield chunk
                
            self.latch(time)
            
        yield from self.emit()

class ConvoFunc:
    '''
    A closure around kernel, pad, filter and convolution functions.
    '''

    def __init__(self, kernel, pad_func=zero_pad, filt_func=None, invert=False):
        '''
        Construct a convo func that convolves a chunk with the kernel after
        applying a pad function.  If filt_func given it is multiplied as part of
        the convolution.
        '''
        self.kernel = kernel
        self.pad_func = pad_func
        self.filt_func = filt_func
        self.invert = invert

    def __call__(self, chunk):
        return convo_func(chunk, self.kernel, self.pad_func, self.filt_func, self.invert)


class PostOverlap:
    '''
    A node that handles a transform that enlarges the size of a chunk by
    appending samples.
    '''
    def __init__(self, transform, chunk_size=None):
        '''
        Apply the transform callable to chunks.

        If given, chunk_size specifies the output chunk size.  Else, the size of
        the first chunk consumed will specify the output chunk size.

        Any additional samples at the end of the transformed, enlarged array are
        held and added to the start of the next enlarged array.

        When the transform is a tail-padded convolution, this implements the
        "overlap-add" method.
        '''
        self.transform = transform
        self.chunk_size = chunk_size

    def __call__(self, chunk_source):
        '''
        Emit transformed chunks.
        '''
        # This holds the tail of current chunk to be added to head of subsequent
        # chunk.
        save = np.zeros((0,), dtype=float)
        for chunk in chunk_source:
            
            if self.chunk_size is None:
                self.chunk_size = chunk.size

            enlarged = self.transform(chunk)

            # Add accrued overlap taking care that it may be smaller or larger
            # than the chunk size
            add_size = min(save.size, self.chunk_size)
            enlarged[:add_size] += save[:add_size]

            # Makes either zero size or pops one chunk worth
            save = save[add_size:]

            done = enlarged[:self.chunk_size]
            tail = enlarged[self.chunk_size:]

            if save.size > tail.size:
                save[:tail.size] += tail
            else:
                tail[:save.size] += save
                save = tail

            yield done

class PreOverlap:
    '''
    A node that handles a transform that enlarges the size of a chunk by
    preppending samples.
    '''
    def __init__(self, transform, chunk_size=None):
        '''
        Apply the transform callable to chunks.

        If given, chunk_size specifies the output chunk size.  Else, the size of
        the first chunk consumed will specify the output chunk size.

        The prior transformed array is held so that the beginning of the next
        transformed array can be added to its end.

        When the transform is a head-padded deconvolution, this implements the
        equivalent to "overlap-add convolution".
        '''
        self.transform = transform
        self.chunk_size = chunk_size

    def __call__(self, chunk_source):
        '''
        Emit transformed chunks.
        '''
        # This holds the prior chunk to be added to the tail of subsequent
        # chunk.
        last = None
        for chunk in chunk_source:
            
            if self.chunk_size is None:
                self.chunk_size = chunk.size

            enlarged = self.transform(chunk)
            
            # Calculate the size of the prepended overlap region
            overlap_size = enlarged.size - self.chunk_size
            
            if overlap_size < 0:
                 raise ValueError(f"Transform output size {enlarged.size} is smaller than expected chunk size {self.chunk_size}")

            if last is None:
                # Prime the prior (last) chunk with the non pre-padded part of the first result.
                # This chunk is not yielded yet.
                last = enlarged[overlap_size:]
                continue

            # Add the overlap (head of enlarged) to the tail of the previous chunk (last)
            last[-overlap_size:] += enlarged[:overlap_size]
            
            done = last
            
            # The new 'last' is the non-overlap part of the current enlarged chunk
            last = enlarged[overlap_size:]
            
            yield done
            
        # Yield the final buffered chunk
        if last is not None:
            yield last
