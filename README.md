# Oil & Gas Deep Learning — True Layered Architecture

Run with:

```powershell
py -m streamlit run app.py
```

Layer ownership:

- `app.py`: starts the app only.
- `core/`: configuration, navigation, state isolation.
- `pages/`: all Streamlit UI page ownership.
- `services/`: preprocessing, training, evaluation, prediction workflow helpers.
- `models/`: ANN/CNN/LSTM model-specific algorithms and model builders only.
- `visualization/`: plotting functions.
- `storage/`: Save/Load and project persistence.
