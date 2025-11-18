import numpy as np
from change_point_algorithms.parsing.run_online import online_model_results_to_polars_v0, online_model_results_to_polars_v1
from change_point_algorithms.DetectionAlgorithm import DetectionAlgorithmV2, ModelType
from polars.testing import assert_frame_equal

def make_time():
    """ """
    return np.linspace(0.0, 0.55, 100)

def make_results():
    return [
        (DetectionAlgorithmV2(ModelType.CUSUM, 'cusum 1', None), [(0.0, 0.5)], [(0.5, 0.55)]),
        (DetectionAlgorithmV2(ModelType.BOCPD, 'bocpd 1', None), [(0.0, 0.25), (0.45, 0.5)], [(0.25, 0.45), (0.5, 0.55)])
    ]

def test_online_model_results_to_polars_v0_equiv_to_v1():
    """ """
    results = make_results()
    time = make_time()
    df_v0 = online_model_results_to_polars_v0(time, results)
    df_v1 = online_model_results_to_polars_v1(time, results)
    assert_frame_equal(df_v0, df_v1)
    # assert df_v0.equals(df_v1), print(df_v0, df_v1, sep='\n')