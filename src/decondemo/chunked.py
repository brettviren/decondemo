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

class ExpoTime:
    def __init__(self, rate=1.0):
        self.rate = rate
    def __call__(self):
        return random.expovariate(self.rate)

class UniformTime:
    def __init__(self, rate=1.0):
        self.rate = rate
    def __call__(self):
        return 1.0/self.rate

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
        while True:
            if not self.limit:
                return
            self.limit -= 1
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
        return int( (time - self.now ) / self.sample_period )

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
            
