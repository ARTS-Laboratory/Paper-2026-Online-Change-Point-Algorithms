import warnings

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    ...


def detection_to_intervals_for_generator_v1(
        time_vec, begin, model_generator, start_offset=0):
    """ Convert detections from generator to time intervals

        This version is for detecting when a measure deviates from expected.
    """
    shock = False
    shocks = list()
    nonshocks = list()
    for idx, is_change in enumerate(model_generator, start=start_offset):
        if is_change and not shock:
            nonshocks.append((time_vec[begin], time_vec[idx]))
            shock = True
            begin = idx
        elif not is_change and shock:
            shocks.append((time_vec[begin], time_vec[idx]))
            shock = False
            begin = idx
    if shock:
        shocks.append((time_vec[begin], time_vec[-1]))
    else:
        nonshocks.append((time_vec[begin], time_vec[-1]))
    return shocks, nonshocks


def detection_to_intervals_for_generator_v1_with_progress(
        time_vec, begin, model_generator, iter_len, start_offset=0):
    """ Convert detections from generator to time intervals."""
    try:
        my_generator = tqdm(model_generator, total=iter_len)
        return detection_to_intervals_for_generator_v1(time_vec, begin, my_generator, start_offset)
    except NameError:
        warnings.warn('tqdm module not found in environment. Running without progress bars.')
        return detection_to_intervals_for_generator_v1(time_vec, begin, model_generator, start_offset)
