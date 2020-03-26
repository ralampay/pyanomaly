import numpy as np

def create_histogram(data, num_bins=100, step=-1):
    min_bin = np.min(data)
    max_bin = np.max(data) + min_bin

    if step < 0:
        step    = (max_bin - min_bin) / num_bins

    bins    = np.arange(min_bin, max_bin, step)

    (hist, bins) = np.histogram(data, bins=bins)

    return (hist,bins)
