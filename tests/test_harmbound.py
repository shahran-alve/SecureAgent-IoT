import copy

import yaml

from secureagent_iot import harmbound as HB
from secureagent_iot.simulator import build_registry

CFG = yaml.safe_load(open("configs/default.yaml"))
REG = build_registry()


def _cfg(fleet=0, corr=0):
    c = copy.deepcopy(CFG)
    c["policy"]["fleet_budget_seconds"] = fleet
    c["policy"]["fleet_corroboration_devices"] = corr
    return c


def test_greedy_attains_device_bound_and_never_exceeds_it():
    c, H = _cfg(), 4 * 3600
    rig = HB.greedy_campaign(c, REG, HB.LOOKALIKES[:2], H)
    b = HB.bound_seconds(c, 2, H)["bound"]
    assert HB.downtime_within(rig.granted, H) == b


def test_fleet_bound_holds_without_shared_indicator():
    c, H = _cfg(1800, 2), 3 * 3600
    rig = HB.greedy_campaign(c, REG, HB.LOOKALIKES[:6], H)
    assert HB.downtime_within(rig.granted, H) == HB.bound_seconds(c, 6, H)["bound"] == 3 * 1800


def test_shared_indicator_lifts_fleet_cap_but_not_device_budget():
    c, H = _cfg(1800, 2), 2 * 3600
    rig = HB.greedy_campaign(c, REG, HB.LOOKALIKES[:6], H, indicator="update:x")
    assert HB.downtime_within(rig.granted, H) == 6 * 2 * 900


def test_random_campaigns_never_exceed_any_tile():
    for fleet in (0, 1800):
        r = HB.random_campaigns(_cfg(fleet), REG, 25, 3 * 3600, seed=3)
        assert r["max_tile_ratio"] <= 1.0 and r["max_device_ratio"] <= 1.0 and r["max_fleet_ratio"] <= 1.0


def test_smt_tile_lemma():
    assert set(HB.smt_tile_lemma(CFG, n_max=5).result) == {"proved"}
