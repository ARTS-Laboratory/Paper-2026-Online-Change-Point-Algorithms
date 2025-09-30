

def benchmark_generator(gen_fn, *args, **kwargs):
    """ Make generator function with given arguments and consume."""
    model_gen = gen_fn(*args, **kwargs)
    curr_item = None
    for item in model_gen:
        curr_item = item
    return curr_item
