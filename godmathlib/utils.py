import time

__all__ = [
    'compare_time','time_func'
]

def compare_time(new,old,data, iterations = 1, rest=0):
    new_times = []
    old_times = []
    for i in range(iterations):
        new_times.append(time_func(new(data),rest))
        old_times.append(time_func(old(data),rest))
    avg_new = sum(new_times)/iterations
    avg_old = sum(old_times)/iterations
    if(avg_new<avg_old):
        print(f"The new function is faster than the old on average by {avg_old-avg_new} over {iterations} iterations")
        return True
    else:
        print(f"The old method is faster on average by {avg_new-avg_old} over {iterations} iterations")
        return False

def time_func(func,rest):
    start = time.perf_counter()
    func
    finish = time.perf_counter()
    time.sleep(rest)
    return finish-start