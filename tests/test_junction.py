from safecross.junction import load_junction_config, stage_times


def test_walk_inside_parallel_green_costs_nothing():
    j = load_junction_config()
    st = stage_times(17.0, j)
    assert st["a_end"] == j["parallel_green_s"]
    assert st["green_lost"] == 0 and st["cycle_extra"] == 0


def test_long_walk_never_cuts_min_green():
    j = load_junction_config()
    st = stage_times(80.0, j)
    assert st["b_green"] >= j["min_green_cross_s"]
    assert st["cycle_extra"] > 0
