# Evaluator Calibration Protocol

Calibration requires independent human labels for at least ten cases spanning clear pass, borderline, and clear fail tiers. The calibration script reports `not_ready` until the minimum is met; the checked-in seed labels are intentionally `unreviewed` placeholders and do not establish evaluator validity.

For a ready calibration, record pass/fail agreement, rank correlation for the five score dimensions, and primary failure-owner accuracy. Threshold or weight changes must be versioned separately from Harness code and must not inspect the frozen test split.
