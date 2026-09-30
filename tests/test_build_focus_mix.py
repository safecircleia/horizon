import random

from training.scripts.build_focus_mix import copies


def test_subtle_grooming_is_oversampled_and_severe_is_not():
    rng = random.Random(0)
    assert copies({"risk_level": "low", "categories": ["grooming"]}, rng) == 3
    assert copies({"risk_level": "critical", "categories": ["grooming"]}, rng) == 1


def test_sexual_content_always_kept_and_others_sampled():
    rng = random.Random(0)
    assert copies({"risk_level": "high", "categories": ["sexual_content"]}, rng) == 1
    kept = sum(
        copies({"risk_level": "none", "categories": []}, rng) for _ in range(2000)
    )
    assert 300 < kept < 500  # ~20% replay
