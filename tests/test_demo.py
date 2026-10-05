from slate.demo import run


def test_the_toy_end_to_end_on_a_small_world():
    r = run(n_contacts=12, n_shards=2, weeks=3, pretrain_steps=60, epochs=12, n_more=3, quiet=True)
    assert r["hash_equal"] and r["shards_equal"] == r["n_shards"]
    # the stale cache always breaks the receipt; whether its leak also shows in decoding depends on how long training ran
    assert r["bug_equal"] is False and r["profile_leak_exact"] is False
    assert r["erased_answered"] == 0
    assert 0.0 <= r["auc"] <= 1.0 and r["auc_pre"] > r["auc"]
