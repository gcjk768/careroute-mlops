"""[Responsible-AI][MLOps] Real ML layer for CareRoute AI.

A scikit-learn severity classifier trained on synthetic triage data, with:
- real SHAP explanations (`model.predict`),
- a real stratified fairness audit with before/after mitigation, and
- real drift metrics (PSI),
all exposed via `app.ml.model.get_model()`.

This is deliberately separate from the LLM agents: the acuity decision becomes a
concrete, inspectable model you can audit for fairness and drift, while the
deterministic Safety-Override still guarantees red-flag cases are never
under-triaged.
"""
