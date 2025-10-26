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


class TimeSource:
    def __init__(self, rate=1.0, start=0.0):
        self.now = start
        self.rate = rate

    def __call__(self):
        pass
    
