import statistics

class Summary:
    def __init__(self, histogram: list):
        self.mean = statistics.fmean(histogram)
        self.range = (min(histogram), max(histogram))