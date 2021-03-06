from collections import defaultdict


class AverageMeter:
    """Compute and store the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset stats.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update stats by adding a sample.

        Args:
            val (float): a value to be added to the stats.
            n (int, optional): sample count for val. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverageMeter:
    """Compute and store the average and current values in dict.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset stats.
        """
        self.val = defaultdict(float)
        self.avg = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, sample, n=1):
        """Update stats by adding a sample.

        Args:
            sample (dict): dict of the values to be added to the stats.
            n (int, optional): count for the sample. Defaults to 1.
        """
        self.val = sample
        for k, v in sample.items():
            self.sum[k] += v
            self.count[k] += n
        for k, v in self.sum.items():
            self.avg[k] = v / self.count[k]
