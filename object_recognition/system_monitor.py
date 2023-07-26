import threading
import time
import psutil
import numpy as np

class SystemMonitor:
    def __init__(self, interval=1):
        self.interval = interval
        self.cpu_usage = []
        self.memory_usage = []

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.thread.join()

    def get_avg_usage(self):
        cpu_avg = np.mean(self.cpu_usage) if self.cpu_usage else 0
        mem_avg = np.mean(self.memory_usage) if self.memory_usage else 0
        return cpu_avg, mem_avg
