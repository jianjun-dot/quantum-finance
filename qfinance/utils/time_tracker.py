from time import time

class Timer():
    
    def __init__(self):
        self.start_time = time()
        self.latest_updated_time = time()
    
    def time_convert(self, sec):
        if sec > 3600:
            hours = sec // 3600
            sec = sec % 3600
            minutes = sec // 60
            sec = sec % 60
            return "{:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),sec)
        elif sec > 60:
            minutes = sec // 60
            sec = sec % 60
            return "{:0>2}m:{:05.2f}s".format(int(minutes),sec)
        else:
            return "{:05.2f}s".format(sec)
        
    def define_loop_start(self):
        self.latest_updated_time = time()
        
    def track_loop(self, total_loops, current_loop):
        time_elapsed = time() - self.latest_updated_time
        time_per_loop = time_elapsed / current_loop
        time_remaining = time_per_loop * (total_loops - current_loop)
        return self.time_convert(time_elapsed), self.time_convert(time_remaining)
    