from memory_profiling.profile_bocpd import profile_bocpd_from_python, profile_bocpd_from_rust
from memory_profiling.profile_cusum import profile_cusum_alg, profile_cusum_alg_v1
from memory_profiling.profile_em import profile_em_from_python
from memory_profiling.profile_grey_model import profile_grey_model_from_python
from memory_profiling.profile_isolation_forest import profile_isolation_forest_from_python


def profile_memory_for_algs():
    """ """
    print('Memory profile for bocpd python.')
    profile_bocpd_from_python()
    print('Memory profile for bocpd rust.')
    profile_bocpd_from_rust()
    print('Memory profile for cusum v0.')
    profile_cusum_alg()
    print('Memory profile for cusum v0 using deques.')
    profile_cusum_alg_v1()
    print('Memory profile for expectation maximization.')
    profile_em_from_python()
    print('Memory profile for grey systems model.')
    profile_grey_model_from_python()
    print('Memory profile for isolation forest.')
    profile_isolation_forest_from_python()
