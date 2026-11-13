"""Small per-load window controller; asynchronous full-round GPU feedback only."""
class SpeculationPolicy:
    def __init__(self):
        self.trials = {}
        self.costs = {}
        self.current = {}

    @staticmethod
    def bucket(batch):
        return 1 << (max(1, batch)-1).bit_length()

    def choose(self, cap, batch):
        load = self.bucket(batch)
        choices = [k for k in (1, 2, 4, 0) if k <= cap]
        if cap not in choices:
            choices.insert(-1, cap)
        for k in choices:
            if not self.trials.get((load, k), 0):
                self.trials[load, k] = 1
                return k
        tick = sum(self.trials.get((load, k), 0) for k in choices)
        if tick % 32 == 0:
            chosen = choices[(tick // 32) % len(choices)]
        else:
            measured = [k for k in choices if (load, k) in self.costs]
            best = min(measured, key=lambda k: self.costs[load, k]) if measured else choices[0]
            previous = self.current.get(load, best)
            chosen = (previous if previous in measured and
                      self.costs[load, previous] <= self.costs[load, best] * 1.05 else best)
        self.current[load] = chosen
        self.trials[load, chosen] = self.trials.get((load, chosen), 0) + 1
        return chosen

    def observe(self, batch, k, emitted, elapsed_ms):
        if emitted <= 0 or elapsed_ms <= 0:
            return
        key = self.bucket(batch), k
        cost = elapsed_ms / emitted
        self.costs[key] = cost if key not in self.costs else self.costs[key] * 0.8 + cost * 0.2
