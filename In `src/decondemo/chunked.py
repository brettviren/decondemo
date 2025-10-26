src/decondemo/chunked.py
...
    def __call__(self, time_source):
        '''
        Emit chunks of latched time
        '''
        for time in time_source:
            # ... (chunking logic) ...
                
            self.latch(time)
            
        yield self.emit() # <--- PROBLEM HERE
