# models/

`risk_model.joblib` is checked in so the API returns a real `risk_assessment`
out of the box, without requiring a training run first. It is a
`RandomForestClassifier` trained on **synthetic-label** data -- see
`data/generated/README.md` and the root `README.md` for exactly what that
means and doesn't mean.

To regenerate it:

```bash
python -m scripts.generate_training_data
python -m scripts.train_model
```
