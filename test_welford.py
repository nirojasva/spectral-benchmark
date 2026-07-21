import math

class WelfordTest:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.std_dev = 0.0

    def start_statistics(self, new_dist):
        self.n = 1
        self.mean = new_dist
        self.sum_squares = 0.0
        self.std_dev = 0.0

    def update_statistics(self, new_dist):
        self.n += 1
        delta = new_dist - self.mean
        self.mean += delta / self.n
        self.sum_squares += delta * (new_dist - self.mean)
        if self.n > 1:
            self.std_dev = math.sqrt(self.sum_squares / (self.n - 1))
        else:
            self.std_dev = float('nan')

w = WelfordTest()
data = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
w.start_statistics(data[0])
for x in data[1:]:
    w.update_statistics(x)

def calc_std(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

print(f"Computed std dev: {w.std_dev}")
print(f"Exact std dev: {calc_std(data)}")
