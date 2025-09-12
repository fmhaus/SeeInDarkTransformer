import time

# Time estimates using exponential moving average (EMA)
class TimeEstimator:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.reset()
    
    def next_interval(self):
        time_passed = time.time() - self.last_time
        
        if self.ema is None:
            self.ema = time_passed
        else:
            self.ema = self.alpha * time_passed + (1 - self.alpha) * self.ema
        
        self.last_time = time.time()
    
    def reset(self):
        self.begin_time = time.time()
        self.last_time = time.time()
        self.ema = None
    
    def get_ema(self):
        return self.ema
    
    def time_passed_total(self):
        return time.time() - self.begin_time

def format_elapsed_time(elapsed_seconds):
    days, rem = divmod(elapsed_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(days):02}d{int(hours):02}h{int(minutes):02}m{int(seconds):02}s"