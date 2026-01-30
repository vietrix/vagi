import torch

from vagi_core.calibration import ConfidenceCalibrator, brier_score


def test_confidence_calibration_improves_brier() -> None:
    confidence = torch.tensor([0.2, 0.8, 0.6, 0.4], dtype=torch.float32)
    outcomes = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    before = brier_score(confidence, outcomes)

    calibrator = ConfidenceCalibrator().fit(confidence, outcomes, steps=200, lr=0.1)
    calibrated = calibrator.apply(confidence)
    after = brier_score(calibrated, outcomes)

    assert torch.isfinite(calibrated).all()
    assert after <= before + 1e-4
