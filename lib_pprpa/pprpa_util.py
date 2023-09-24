import numpy
import time

def ij2index(r, c, row, col):
    """Get index of a row and column in a square matrix in a lower triangular matrix.

    Args:
        r (int): row index in s square matrix.
        c (int): column index in s square matrix.
        row (int array): row index array of a lower triangular matrix.
        col (int array): column index array of a lower triangular matrix.

    Returns:
        i (int): index in the lower triangular matrix.
    """
    for i in range(len(row)):
        if r == row[i] and c == col[i]:
            return i

    raise ValueError("cannot find the index!")


def inner_product(u, v, oo_dim):
    """Calculate ppRPA inner product.
    product = <Y1,Y2> - <X1,X2>, where X is occ-occ block, Y is vir-vir block.

    Args:
        u (double array): first vector.
        v (double array): second vector
        oo_dim (int): occ-occ block dimension

    Returns:
        inp (double): inner product.
    """
    inp = -numpy.sum(u[:oo_dim] * v[:oo_dim]) + numpy.sum(u[oo_dim:] * v[oo_dim:])
    return inp


# time counting global variables and functions
clock_names = []
clocks = []

def _s_to_hms(t):
    decimal = t - int(t)
    t = int(t)
    seconds = int(t % 60)
    t = (t - seconds) / 60
    minutes = int(t % 60)
    hours = int((t - minutes) / 60)
    hms = "%.2f s" % (seconds + decimal)
    if minutes != 0 or hours != 0:
        hms = ("%d m " % minutes) + hms
    if hours != 0:
        hms = ("%d h " % hours) + hms
    return hms


def start_clock(clock_name):
    assert isinstance(clock_name, str) and clock_name not in clock_names
    clock_names.append(clock_name)
    clocks.append((time.process_time(), time.perf_counter()))
    print("begin %-s." % clock_name, flush=True)


def stop_clock(clock_name):
    assert isinstance(clock_name, str) and clock_name in clock_names
    idx = clock_names.index(clock_name)
    clock_end = (time.process_time(), time.perf_counter())
    cpu_time = _s_to_hms(clock_end[0] - clocks[idx][0])
    wall_time = _s_to_hms(clock_end[1] - clocks[idx][1])
    del clock_names[idx]
    del clocks[idx]

    print("finish %-s." % clock_name)
    print('    CPU time for %s %s, wall time %s\n' % (clock_name, cpu_time, wall_time), flush=True)
