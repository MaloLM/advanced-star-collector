import time


class Timer:

    def __init__(self) -> None:
        self.reset()

    def get_current_duration(self):
        if self.start_ts is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_ts

    def get_formatted_duration(self):
        duration = self.get_current_duration() if self.end_ts is None else self.duration
        return self.format_duration(duration)

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def start(self):
        self.start_ts = time.time()
        self.end_ts = None

    def end(self):
        if self.start_ts is None:
            raise ValueError("Timer has not been started.")
        self.end_ts = time.time()
        self.duration = self.end_ts - self.start_ts
        return self.format_duration(self.duration)

    def reset(self):
        self.start_ts = None
        self.end_ts = None
        self.duration = None

    def restart(self):
        self.reset()
        self.start()
