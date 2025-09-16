import time
import numpy as np
from multiprocessing import Manager

def print_evaluation(timings):
    header = ["section", "percentage", "count", "time-total", "time-avg", "time-min", "time-max", "percentile90", "percentile99"]
    rows = [header]
    sums = []
    for name, times in timings.items():
        if len(times) == 0:
            sums.append(0)
            rows.append([name, "-", "0"] + ["-"]*6)
            continue
        
        NS_TO_SEC = 1e-9
        times = np.array(times, dtype=np.float64)
        sums.append(np.sum(times))
        entries = [name, ""]
        entries.append(f"{len(times)}")
        entries.append(f"{np.sum(times) * NS_TO_SEC:.2f}")
        entries.append(f"{np.mean(times) * NS_TO_SEC:.2f}")
        entries.append(f"{np.min(times) * NS_TO_SEC:.2f}")
        entries.append(f"{np.max(times) * NS_TO_SEC:.2f}")
        entries.append(f"{np.percentile(times, 90) * NS_TO_SEC:.2}")
        entries.append(f"{np.percentile(times, 99) * NS_TO_SEC:.2}")
        rows.append(entries)
    total_sum = np.sum(sums)
    if total_sum != 0:
        for i, sum in enumerate(sums):
            rows[i+1][1] = f"{100 * sum / total_sum:.2f}%"
    widths = [0 for _ in rows[0]]
    for row in rows:
        for j, word in enumerate(row):
            widths[j] = max(widths[j], len(word))
    for row in rows:
        for j, word in enumerate(row):
            print(word.rjust(widths[j]), end=" ")
        print()

class ProcessProfiling():
    def __init__(self, manager, sections):
        self.timings = manager.dict()
        for section in sections:
            self.timings[section] = manager.list()
        
    def print_evaluation(self):
        print_evaluation({name: list(times) for name, times in self.timings.items()})
    
    def append_timings(self, timings):
        for name, times in timings.items():
            self.timings[name].extend(times)
    
    def reset_timings(self):
        for times in self.timings.values():
            times[:] = []

class DummyProfiler():
    def section(self, _):
        pass
    
    def end_section(self):
        pass

    def reset_timings(self):
        pass
        
    def print_evaluation(self):
        pass

    def save_to_profiling(self, _):
        pass
    
class Profiler():
    def __init__(self):
        self.current_section = None
        self.section_time = None
        self.timings = {}
    
    def section(self, name):
        assert self.current_section is None
        self.current_section = name
        self.section_time = time.perf_counter_ns()
    
    def end_section(self):
        assert self.current_section is not None
        time_passed = time.perf_counter_ns() - self.section_time
        
        if self.current_section in self.timings:
            self.timings[self.current_section].append(time_passed)
        else:
            self.timings[self.current_section] = [time_passed]
        
        self.current_section = None

    def reset_timings(self):
        self.timings.clear()

    def print_evaluation(self):
        print_evaluation({name: list(times) for name, times in self.timings.items()})

    def save_to_profiling(self, profiling):
        profiling.append_timings(self.timings)