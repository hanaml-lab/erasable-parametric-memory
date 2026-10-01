import random

import pytest

from slate.mia import auc, bootstrap_ci, two_sided


def test_auc_basics():
    assert auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert auc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert auc([1.0], [1.0]) == 0.5
    assert two_sided(0.2) == 0.8 and two_sided(0.7) == 0.7
    with pytest.raises(ValueError):
        auc([], [1.0])


def test_identical_laws_give_chance_and_the_interval_covers_it():
    rnd = random.Random(0)
    pos = [rnd.gauss(0, 1) for _ in range(300)]
    neg = [rnd.gauss(0, 1) for _ in range(300)]
    a = auc(pos, neg)
    lo, hi = bootstrap_ci(pos, neg, n=200)
    assert abs(a - 0.5) < 0.06 and lo < 0.5 < hi


def test_grouped_bootstrap_is_wider_for_clustered_data():
    rnd = random.Random(1)
    pos, groups = [], []
    for g in range(6):
        shift = rnd.gauss(0, 1)
        for _ in range(10):
            pos.append(shift + rnd.gauss(0, 0.1))
            groups.append(g)
    neg = [rnd.gauss(0, 1) for _ in range(60)]
    lo1, hi1 = bootstrap_ci(pos, neg, n=300)
    lo2, hi2 = bootstrap_ci(pos, neg, n=300, pos_groups=groups)
    assert hi2 - lo2 > hi1 - lo1
