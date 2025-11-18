import tracemalloc

DEFAULT_MEM_UNIT = 'KiB'

def get_scalar_unit(unit: str):
    match unit:
        case 'B':
            scalar = 1
        case 'KiB':
            scalar = 1_024
        case 'MiB':
            scalar = 2**20
        case 'KB':
            scalar = 1e3
        case 'MB':
            scalar = 1e6
        case _:
            raise ValueError(f'Unit {unit} not supported')
    return scalar

# def convert_value(raw_value, scalar):
#     return raw_value / scalar

def profile_model_run(model_gen, unit=None):
    mem_unit = unit if unit is not None else DEFAULT_MEM_UNIT
    mem_scalar = get_scalar_unit(mem_unit)
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After generator current memory usage: {curr / mem_scalar} {mem_unit}, peak of {peak / mem_scalar} {mem_unit}')
    tracemalloc.reset_peak()
    # Take 1 element
    next(model_gen)
    print(tracemalloc.get_traced_memory())
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After calling next once current memory usage: {curr / mem_scalar:.4} {mem_unit}, peak of {peak / mem_scalar:.4} {mem_unit}')
    for item in model_gen:
        continue
    # out = [item for item in model_gen]
    curr, peak = tracemalloc.get_traced_memory()
    print(f'After data collection current memory usage: {curr / mem_scalar:.4} {mem_unit}, peak of {peak / mem_scalar:.4} {mem_unit}')

# def memory_peaks_for_gen(func):
#     """ A decorator for tracking memory peaks for a function."""
#     def wrapper(*args, **kwargs):
#         # Before
#         print('\n')
#         tracemalloc.start(20)
#         curr, peak = tracemalloc.get_traced_memory()
#         print(f'Current memory usage: {curr} B, peak of {peak} B')
#         # func
#         # After
#         tracemalloc.stop()
