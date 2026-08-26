# Optional model-layer optimization

Preference export is deliberately downstream of Harness stability. `export_preference_pairs.py` refuses to write data until evaluator calibration is ready, a train-improving/dev-safe Harness candidate is accepted, and test reproducibility is recorded. Plan pairs and trajectory repair pairs are written to separate JSONL files and must be evaluated independently from the Harness score.
