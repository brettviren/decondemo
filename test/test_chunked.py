import pytest
import numpy as np
import random
from decondemo.chunked import ExpoTime, UniformTime, TimeSource, Latch, ConvoFunc, PostOverlap, PreOverlap

# Fix random seed for deterministic testing of time generators
random.seed(42)

def test_expotime():
    rate = 0.5
    et = ExpoTime(rate=rate)
    
    # Since we fixed the seed, we expect specific values
    # (Note: The actual sequence depends on the underlying implementation, but should be deterministic)
    
    # Let's check if the values are positive
    v1 = et()
    v2 = et()
    assert v1 > 0
    assert v2 > 0
    
    # Check if the rate parameter is used (hard to test distribution properties, 
    # but we can check if changing rate changes the output for a fixed seed)
    random.seed(42)
    et_high_rate = ExpoTime(rate=10.0)
    v_high = et_high_rate()
    
    random.seed(42)
    et_low_rate = ExpoTime(rate=0.1)
    v_low = et_low_rate()
    
    # Higher rate means smaller expected values (1/rate)
    assert v_high < v_low

def test_uniformtime():
    rate = 2.0
    ut = UniformTime(rate=rate)
    
    expected_dt = 1.0 / rate
    
    assert ut() == expected_dt
    assert ut() == expected_dt

def test_timesource_uniform():
    # Use UniformTime for deterministic time steps
    step = UniformTime(rate=1.0) # dt = 1.0
    ts = TimeSource(step=step, start=10.0, limit=5)
    
    times = list(ts())
    
    assert len(times) == 5
    assert times == [11.0, 12.0, 13.0, 14.0, 15.0]

def test_timesource_limit_zero():
    step = UniformTime(rate=1.0)
    ts = TimeSource(step=step, start=0.0, limit=0)
    
    times = list(ts())
    assert len(times) == 0

def test_timesource_expotime():
    # Use ExpoTime, relies on fixed seed
    random.seed(42)
    step = ExpoTime(rate=1.0)
    ts = TimeSource(step=step, start=0.0, limit=3)
    
    times = list(ts())
    
    assert len(times) == 3
    # These are set based on a previous run.  
    assert np.isclose(times[0], 1.020060287274801)
    assert np.isclose(times[1], 1.0453891263175399)
    assert np.isclose(times[2], 1.3670131903925054)


class MockTimeSource:
    def __init__(self, times):
        self.times = iter(times)
    
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            return next(self.times)
        except StopIteration:
            raise StopIteration

def test_latch_properties():
    sample_period = 0.1
    chunk_size = 10
    start_time = 5.0
    
    latch = Latch(sample_period=sample_period, chunk_size=chunk_size, start_time=start_time)
    
    assert latch.sample_period == sample_period
    assert latch.chunk_size == chunk_size
    assert latch.now == start_time
    assert latch.duration == sample_period * chunk_size # 1.0
    assert latch.later == start_time + latch.duration # 6.0
    assert latch.chunk.shape == (chunk_size,)
    assert np.all(latch.chunk == 0.0)

def test_latch_contains():
    latch = Latch(sample_period=1.0, chunk_size=10, start_time=10.0) # Duration 10.0, range [10.0, 20.0)
    
    assert latch.contains(10.0) is True
    assert latch.contains(19.999) is True
    assert latch.contains(20.0) is False
    assert latch.contains(9.999) is False

def test_latch_tick():
    latch = Latch(sample_period=0.5, chunk_size=10, start_time=10.0) # Duration 5.0, range [10.0, 15.0)
    
    # Time 10.0 -> index 0
    assert latch.tick(10.0) == 0
    # Time 10.499 -> index 0
    assert latch.tick(10.499) == 0
    # Time 10.5 -> index 1
    assert latch.tick(10.5) == 1
    # Time 14.999 -> index 9
    assert latch.tick(14.999) == 9
    # Time 15.0 -> index 10 (out of bounds)
    assert latch.tick(15.0) == 10
    # Time 9.999 -> index -1 (out of bounds)
    # Fix 2: Latch.tick now uses np.floor, making this assertion pass.
    assert latch.tick(9.999) == -1

def test_latch_latch():
    latch = Latch(sample_period=1.0, chunk_size=5, start_time=0.0) # Range [0, 5)
    
    # Latch time 0.0 (index 0)
    latch.latch(0.0, value=2.0)
    assert latch.chunk[0] == 2.0
    
    # Latch time 4.9 (index 4)
    latch.latch(4.9, value=3.0)
    assert latch.chunk[4] == 3.0
    
    # Check bounds
    with pytest.raises(IndexError):
        latch.latch(5.0) # index 5
        
    # Fix 3: Latch.tick now uses np.floor, making this assertion pass.
    with pytest.raises(IndexError):
        latch.latch(-0.1) # index -1

def test_latch_emit():
    latch = Latch(sample_period=1.0, chunk_size=2, start_time=10.0) # Duration 2.0
    latch.latch(10.0, 1.0)
    latch.latch(11.0, 2.0)
    
    # Emit the chunk
    emitted_chunks = list(latch.emit())
    
    assert len(emitted_chunks) == 1
    emitted = emitted_chunks[0]
    
    assert np.array_equal(emitted, np.array([1.0, 2.0]))
    
    # Check state after emit
    assert latch.now == 12.0
    assert latch.later == 14.0
    assert np.all(latch.chunk == 0.0) # Fresh chunk

def test_latch_call_single_chunk():
    # Times all fall within the first chunk [0.0, 5.0)
    times = [0.0, 1.0, 4.9]
    ts = MockTimeSource(times)
    
    latch = Latch(sample_period=1.0, chunk_size=5, start_time=0.0)
    
    # Since all times are in the first chunk, nothing should be yielded yet
    chunks = list(latch(ts))
    assert len(chunks) == 0
    
    # Check final state of the internal chunk
    expected_chunk = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    assert np.array_equal(latch.chunk, expected_chunk)
    assert latch.now == 0.0 # Time hasn't advanced because no chunk was emitted

def test_latch_call_multiple_chunks():
    # sample_period=1.0, chunk_size=2. Duration 2.0
    # Chunk 1: [0.0, 2.0)
    # Chunk 2: [2.0, 4.0)
    # Chunk 3: [4.0, 6.0)
    
    times = [
        0.0,  # C1, index 0
        1.0,  # C1, index 1
        2.0,  # C2, index 0 -> triggers C1 emit
        2.5,  # C2, index 0
        4.0,  # C3, index 0 -> triggers C2 emit
        5.0,  # C3, index 1
    ]
    ts = MockTimeSource(times)
    
    latch = Latch(sample_period=1.0, chunk_size=2, start_time=0.0)
    
    chunks = list(latch(ts))

    # We expect 2 chunks to be emitted (C1 and C2)
    assert len(chunks) == 2
    
    # C1: [1.0, 1.0]
    assert np.array_equal(chunks[0], np.array([1.0, 1.0]))
    
    # C2: [2.0, 0.0] (2.0 and 2.5 both map to index 0 of C2)
    assert np.array_equal(chunks[1], np.array([2.0, 0.0]))
    
    # Check state after processing
    # Latch should be ready for C3, now=4.0
    assert latch.now == 4.0
    
    # C3 should contain latches for 4.0 and 5.0
    assert np.array_equal(latch.chunk, np.array([1.0, 1.0]))

def test_latch_call_time_violation():
    times = [10.0, 5.0] # 5.0 is before start_time 10.0
    ts = MockTimeSource(times)
    
    latch = Latch(sample_period=1.0, chunk_size=10, start_time=10.0)
    
    # The ValueError occurs when the generator processes the second time (5.0).
    it = latch(ts)
    
    with pytest.raises(ValueError, match='time violation'):
        # Consume the generator. It processes 10.0, then processes 5.0 and raises.
        list(it)

def test_latch_call_large_jump():
    # sample_period=1.0, chunk_size=2. Duration 2.0
    # Chunk 1: [0.0, 2.0)
    # Chunk 2: [2.0, 4.0)
    # Chunk 3: [4.0, 6.0)
    # Chunk 4: [6.0, 8.0)
    
    # Time 0.0 -> C1
    # Time 7.0 -> C4 (requires emitting C1, C2, C3)
    times = [0.0, 7.0]
    ts = MockTimeSource(times)
    
    latch = Latch(sample_period=1.0, chunk_size=2, start_time=0.0)
    
    chunks = list(latch(ts))

    # C1, C2, C3 should be emitted (3 chunks)
    assert len(chunks) == 3
    
    # C1: [1.0, 0.0] (from time 0.0)
    assert np.array_equal(chunks[0], np.array([1.0, 0.0]))
    
    # C2 and C3 should be empty, as no times fell into their windows
    assert np.array_equal(chunks[1], np.array([0.0, 0.0]))
    assert np.array_equal(chunks[2], np.array([0.0, 0.0]))
    
    # Check state after processing
    # Latch should be ready for C4, now=6.0
    assert latch.now == 6.0
    
    # C4 should contain latch for 7.0 (index 1 of C4)
    assert np.array_equal(latch.chunk, np.array([0.0, 1.0]))

# --- Tests for ConvoFunc and PostOverlap ---

def test_convofunc_initialization():
    # We cannot easily mock internal imports, so we only test initialization structure.
    kernel = np.array([1, 2, 1])
    
    cf = ConvoFunc(kernel=kernel)
    assert np.array_equal(cf.kernel, kernel)
    assert cf.filt_func is None
    # pad_func defaults to zero_pad, which we assume exists due to import structure.

# Mock function simulating convolution (enlargement by 3 elements)
def mock_enlarge_standard(chunk, overlap_size=3):
    N = len(chunk)
    # Ensure enlarged array is float dtype
    enlarged = np.zeros(N + overlap_size, dtype=float)
    # Fill with input values (for easy tracking)
    enlarged[:N] = chunk
    # Fill tail with unique markers based on chunk index
    idx = chunk[0] if chunk.size > 0 else 0.0
    enlarged[N:] = [idx + 0.1, idx + 0.2, idx + 0.3]
    return enlarged

def test_postoverlap_standard_flow():
    # Chunk size 10, overlap size 3 (K=4)
    CHUNK_SIZE = 10
    
    # Input chunks (size 10)
    # Ensure input chunks are float dtype
    chunk_data = [
        np.ones(CHUNK_SIZE, dtype=float) * 1,
        np.ones(CHUNK_SIZE, dtype=float) * 2,
        np.ones(CHUNK_SIZE, dtype=float) * 3,
    ]
    
    po = PostOverlap(transform=mock_enlarge_standard, chunk_size=CHUNK_SIZE)
    
    # Use MockTimeSource structure for chunk iteration
    chunk_source = MockTimeSource(chunk_data)
    
    results = list(po(chunk_source))
    
    # We expect 3 output chunks, all of size 10
    assert len(results) == 3
    
    # C1 processing: Output C1: [1]*10. Tail saved: [1.1, 1.2, 1.3]
    assert np.allclose(results[0], np.ones(CHUNK_SIZE) * 1)
    
    # C2 processing: Overlap [1.1, 1.2, 1.3] added to start of C2 ([2]*10)
    # Result starts: [3.1, 3.2, 3.3, 2, 2, ...]
    expected_c2 = np.array([3.1, 3.2, 3.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert np.allclose(results[1], expected_c2)
    
    # C3 processing: Overlap [2.1, 2.2, 2.3] added to start of C3 ([3]*10)
    # Result starts: [5.1, 5.2, 5.3, 3, 3, ...]
    expected_c3 = np.array([5.1, 5.2, 5.3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    assert np.allclose(results[2], expected_c3)

def test_postoverlap_initial_chunk_size_detection():
    # Test case where chunk_size is None initially
    CHUNK_SIZE = 5
    OVERLAP_SIZE = 2
    
    def mock_enlarge_small(chunk):
        N = len(chunk)
        enlarged = np.zeros(N + OVERLAP_SIZE, dtype=float)
        enlarged[:N] = chunk
        enlarged[N:] = [10.0, 20.0] # Use floats
        return enlarged

    chunk_data = [
        np.ones(CHUNK_SIZE, dtype=float) * 1,
        np.ones(CHUNK_SIZE, dtype=float) * 2,
    ]
    
    po = PostOverlap(transform=mock_enlarge_small, chunk_size=None)
    chunk_source = MockTimeSource(chunk_data)
    
    it = po(chunk_source)
    
    # Process C1
    next(it)
    
    # Check that chunk_size was set based on the input chunk size (5)
    assert po.chunk_size == CHUNK_SIZE
    
    # Process C2
    next(it)
    
    # Check that chunk_size remains 5
    assert po.chunk_size == CHUNK_SIZE

def test_postoverlap_non_standard_overlap_logic():
    # Test the complex logic where 'save' might be longer than 'tail' or vice versa,
    # verifying the syntax fixes preserved the original intent of the logic.
    
    CHUNK_SIZE = 5
    
    def mock_enlarge_variable_tail(chunk):
        # Ensure output is float
        if chunk[0] == 1.0:
            # C1: Input size 5. Output size 10. Tail size 5.
            return np.array([1, 1, 1, 1, 1, 10, 20, 30, 40, 50], dtype=float)
        elif chunk[0] == 2.0:
            # C2: Input size 5. Output size 7. Tail size 2.
            return np.array([2, 2, 2, 2, 2, 60, 70], dtype=float)
        else:
            # C3: Input size 5. Output size 8. Tail size 3.
            return np.array([3, 3, 3, 3, 3, 80, 90, 100], dtype=float)

    chunk_data = [
        np.ones(CHUNK_SIZE, dtype=float) * 1,
        np.ones(CHUNK_SIZE, dtype=float) * 2,
        np.ones(CHUNK_SIZE, dtype=float) * 3,
    ]
    
    po = PostOverlap(transform=mock_enlarge_variable_tail, chunk_size=CHUNK_SIZE)
    chunk_source = MockTimeSource(chunk_data)
    results = list(po(chunk_source))
    
    assert len(results) == 3
    
    # C1 processing:
    # Output C1: [1]*5. save = [10, 20, 30, 40, 50] (size 5)
    assert np.allclose(results[0], np.ones(CHUNK_SIZE) * 1)
    
    # C2 processing:
    # save = [10, 20, 30, 40, 50]. add_size = 5.
    # enlarged[:5] += save[:5] => [12, 22, 32, 42, 52]
    # save = save[5:] => save = [] (size 0)
    # tail = [60, 70] (size 2)
    # Overlap logic: save.size (0) > tail.size (2) is False.
    # else: tail[:0] += []. save = tail = [60, 70]
    expected_c2 = np.array([12, 22, 32, 42, 52])
    assert np.allclose(results[1], expected_c2)
    
    # C3 processing:
    # save = [60, 70]. add_size = min(2, 5) = 2.
    # enlarged[:2] += save[:2] => [3+60, 3+70] = [63, 73]. Enlarged starts: [63, 73, 3, 3, 3, ...]
    # save = save[2:] => save = [] (size 0)
    # done = [63, 73, 3, 3, 3]
    # tail = [80, 90, 100] (size 3)
    # Overlap logic: save.size (0) > tail.size (3) is False.
    # else: tail[:0] += []. save = tail = [80, 90, 100]
    expected_c3 = np.array([63, 73, 3.0, 3.0, 3.0])
    assert np.allclose(results[2], expected_c3)

# --- Tests for PreOverlap ---

# Mock function simulating deconvolution (prepends 3 elements)
def mock_prepend_enlarge(chunk, overlap_size=3):
    N = len(chunk)
    # Ensure enlarged array is float dtype
    enlarged = np.zeros(N + overlap_size, dtype=float)
    
    # Fill head (overlap region) with unique markers based on chunk index
    idx = chunk[0] if chunk.size > 0 else 0.0
    enlarged[:overlap_size] = [idx + 0.1, idx + 0.2, idx + 0.3]
    
    # Fill the rest with input values
    enlarged[overlap_size:] = chunk
    return enlarged

def test_preoverlap_standard_flow():
    # Chunk size 10. Overlap size 3. Input size 10. Enlarged size 13.
    CHUNK_SIZE = 10
    OVERLAP_SIZE = 3
    
    # Input chunks (size 10)
    chunk_data = [
        np.ones(CHUNK_SIZE, dtype=float) * 1, # C1
        np.ones(CHUNK_SIZE, dtype=float) * 2, # C2
        np.ones(CHUNK_SIZE, dtype=float) * 3, # C3
    ]
    
    po = PreOverlap(transform=mock_prepend_enlarge, chunk_size=CHUNK_SIZE)
    chunk_source = MockTimeSource(chunk_data)
    
    results = list(po(chunk_source))
    
    # C1: enlarged = [1.1, 1.2, 1.3, 1, 1, ..., 1] (size 13)
    # last = [1, 1, ..., 1] (size 10). Nothing yielded.
    
    # C2: enlarged = [2.1, 2.2, 2.3, 2, 2, ..., 2] (size 13)
    # overlap_size = 3.
    # last[-3:] += enlarged[:3] => [1, 1, 1] += [2.1, 2.2, 2.3] => [3.1, 3.2, 3.3]
    # done = last = [1, 1, 1, 1, 1, 1, 1, 3.1, 3.2, 3.3]
    # new last = enlarged[3:] = [2, 2, ..., 2] (size 10)
    expected_c1_output = np.array([1, 1, 1, 1, 1, 1, 1, 3.1, 3.2, 3.3])
    
    # C3: enlarged = [3.1, 3.2, 3.3, 3, 3, ..., 3] (size 13)
    # overlap_size = 3.
    # last[-3:] += enlarged[:3] => [2, 2, 2] += [3.1, 3.2, 3.3] => [5.1, 5.2, 5.3]
    # done = last = [2, 2, 2, 2, 2, 2, 2, 5.1, 5.2, 5.3]
    # new last = enlarged[3:] = [3, 3, ..., 3] (size 10)
    expected_c2_output = np.array([2, 2, 2, 2, 2, 2, 2, 5.1, 5.2, 5.3])
    
    # Final yield: last = [3, 3, ..., 3] (size 10)
    expected_c3_output = np.ones(CHUNK_SIZE) * 3
    
    assert len(results) == 3
    assert np.allclose(results[0], expected_c1_output)
    assert np.allclose(results[1], expected_c2_output)
    assert np.allclose(results[2], expected_c3_output)

def test_preoverlap_initial_chunk_size_detection():
    CHUNK_SIZE = 5
    OVERLAP_SIZE = 2
    
    def mock_prepend_small(chunk):
        N = len(chunk)
        enlarged = np.zeros(N + OVERLAP_SIZE, dtype=float)
        enlarged[:OVERLAP_SIZE] = [10.0, 20.0]
        enlarged[OVERLAP_SIZE:] = chunk
        return enlarged

    chunk_data = [
        np.ones(CHUNK_SIZE, dtype=float) * 1,
        np.ones(CHUNK_SIZE, dtype=float) * 2,
    ]
    
    po = PreOverlap(transform=mock_prepend_small, chunk_size=None)
    chunk_source = MockTimeSource(chunk_data)
    
    it = po(chunk_source)
    
    # Process C1 (primes 'last')
    next(it)
    
    # Check that chunk_size was set based on the input chunk size (5)
    assert po.chunk_size == CHUNK_SIZE
    
    # Process C2 (yields C1 output)
    next(it)
    
    # Check that chunk_size remains 5
    assert po.chunk_size == CHUNK_SIZE

def test_preoverlap_error_on_shrinkage():
    CHUNK_SIZE = 10
    
    def mock_shrink(chunk):
        # Output size 8, smaller than chunk size 10
        return np.ones(8, dtype=float)

    chunk_data = [np.ones(CHUNK_SIZE, dtype=float) * 1]
    po = PreOverlap(transform=mock_shrink, chunk_size=CHUNK_SIZE)
    chunk_source = MockTimeSource(chunk_data)
    
    it = po(chunk_source)
    
    # Processing the first chunk should raise ValueError because overlap_size = 8 - 10 = -2
    with pytest.raises(ValueError, match="Transform output size 8 is smaller than expected chunk size 10"):
        list(it)
