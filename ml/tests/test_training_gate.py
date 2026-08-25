from crrc_vision.training import TrainingReadiness


def test_training_gate_rejects_unreviewed_annotations():
    report = TrainingReadiness(images=482, accepted=0, rejected=0, unreviewed=482)

    assert report.can_train is False
    assert "unreviewed" in report.reasons[0]


def test_training_gate_rejects_low_precision_prelabels():
    report = TrainingReadiness(images=60, accepted=61, rejected=39, unreviewed=0)

    assert report.can_train is False
    assert any("precision" in reason for reason in report.reasons)


def test_training_gate_accepts_reviewed_dataset():
    report = TrainingReadiness(images=80, accepted=75, rejected=5, unreviewed=0)

    assert report.can_train is True
    assert report.reasons == []
