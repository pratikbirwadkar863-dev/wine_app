# Wine Prediction App

    pip install -r requirements.txt
    python train_model.py      # only if you don't already have best_wine_model.pkl + wine_data.csv
    streamlit run app.py

- Images live in `assets/`: `hero.jpg` (header) and `class_0/1/2.jpg` (glass illustrations). Replace any of them with your own photos; regenerate the defaults with `make_hero.py` and `make_glasses.py`.
- If you use your own notebook's model, copy `best_wine_model.pkl` and `wine_data.csv` next to `app.py`.
  The model must accept the 13 sklearn wine features as a DataFrame.
