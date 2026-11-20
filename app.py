"""
Wine Prediction App (Streamlit)

Run with:
    streamlit run app.py

Photo header: assets/hero.jpg (swap in your own image any time).
"""
import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_wine_model.pkl"
DATA_PATH = BASE_DIR / "wine_data.csv"
HERO_PATH = BASE_DIR / "assets" / "hero.jpg"
ASSETS = BASE_DIR / "assets"

CLASS_NAMES = {0: "Class 0", 1: "Class 1", 2: "Class 2"}

GROUPS = {
    "Acidity and minerals": ["malic_acid", "ash", "alcalinity_of_ash", "magnesium"],
    "Body": ["alcohol", "proline"],
    "Phenols": ["total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins"],
    "Colour and clarity": ["color_intensity", "hue", "od280/od315_of_diluted_wines"],
}

# Units are the ones commonly cited for the UCI/sklearn wine data; the dataset itself does not document most of them.
MEASURES = {
    "alcohol": ("% vol", "Alcohol by volume"),
    "malic_acid": ("g/L", "Malic acid, the sharp, green-apple acid"),
    "ash": ("g/L", "Ash left after evaporation, a mineral content indicator"),
    "alcalinity_of_ash": ("mEq/L", "Alkalinity of the ash"),
    "magnesium": ("mg/L", "Magnesium"),
    "total_phenols": ("mg/L", "All phenolic compounds together"),
    "flavanoids": ("mg/L", "Flavanoid phenols, linked to bitterness and colour"),
    "nonflavanoid_phenols": ("mg/L", "Non-flavanoid phenols"),
    "proanthocyanins": ("mg/L", "Proanthocyanins (condensed tannins)"),
    "color_intensity": ("AU", "Colour intensity, an absorbance reading"),
    "hue": ("ratio", "Colour hue, a ratio of absorbance readings"),
    "od280/od315_of_diluted_wines": ("ratio", "OD280/OD315 ratio of diluted wine, a protein-related reading"),
    "proline": ("mg/L", "Proline, the main amino acid in grape must"),
}

st.set_page_config(page_title="Wine Prediction", page_icon="🍷", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.5rem; max-width: 1150px; }

.hero {
    border-radius: 6px; padding: 4rem 2.4rem; color: #FAF6F0;
    background-color: #4A1520; background-size: cover; background-position: center;
    margin-bottom: 1.5rem;
}
.hero h1 { color: #FAF6F0; font-size: 2.5rem; margin: 0 0 .5rem 0; }
.hero p  { margin: 0; opacity: .9; max-width: 560px; line-height: 1.5; }

.result-box {
    background: #FFFFFF; border: 1px solid #E2D6C6; border-left: 5px solid #7B1E2B;
    border-radius: 4px; padding: 1.2rem 1.4rem;
}
.result-box .label { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; color: #8A7468; }
.result-box .big   { font-family: 'Playfair Display', serif; font-size: 2.2rem; color: #7B1E2B; line-height: 1.1; }
.result-box .sub   { color: #5C4B45; margin-top: .2rem; }

.scale { position: relative; height: 8px; border-radius: 4px; margin: 1.6rem 0 .3rem 0;
         background: linear-gradient(90deg, #D9B8B0, #A8404F, #4A1520); }
.scale .dot { position: absolute; top: -5px; width: 18px; height: 18px; border-radius: 50%;
              background: #FAF6F0; border: 3px solid #2A1F1D; transform: translateX(-50%); }
.ticks { display: flex; justify-content: space-between; font-size: .8rem; color: #8A7468; }

.group-title { font-family: 'Playfair Display', serif; font-size: 1.15rem; color: #7B1E2B;
               border-bottom: 1px solid #E2D6C6; padding-bottom: .25rem; margin: 1.2rem 0 .4rem 0; }
.gallery { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: 1rem; }
.card { background: #fff; border: 1px solid #E2D6C6; border-radius: 4px; overflow: hidden; opacity: .55;
        transition: opacity .2s; text-align: center; padding-bottom: .6rem; }
.card.on { opacity: 1; border: 2px solid #7B1E2B; box-shadow: 0 2px 10px rgba(123,30,43,.18); }
.card img { width: 100%; display: block; }
.card .cn { font-family: 'Playfair Display', serif; font-size: 1.05rem; margin-top: .5rem; }
.card .ct { font-size: .78rem; color: #8A7468; padding: 0 .4rem; }
div[data-testid="stSidebar"] { border-right: 1px solid #E2D6C6; }
button[kind="primary"] { border-radius: 3px; font-weight: 600; }
</style>
"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def hero(title, text):
    style = ""
    if HERO_PATH.exists():
        b64 = base64.b64encode(HERO_PATH.read_bytes()).decode()
        style = (
            "background-image: linear-gradient(rgba(42,10,16,.45), rgba(42,10,16,.55)), "
            f"url('data:image/jpeg;base64,{b64}');"
        )
    st.markdown(
        f'<div class="hero" style="{style}"><h1>{title}</h1><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def img_b64(path):
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def class_gallery(nearest, class_means, overall_mean, overall_std):
    """Three class cards with a glass picture; the predicted one is highlighted."""
    cards = []
    for k, name in CLASS_NAMES.items():
        z = ((class_means.loc[k] - overall_mean) / overall_std).sort_values(key=abs, ascending=False)
        traits = []
        for feat in z.index[:2]:
            word = "high" if z[feat] > 0 else "low"
            traits.append(f"{word} {feat.replace('_', ' ')}")
        cls = "card on" if k == nearest else "card"
        b64 = img_b64(ASSETS / f"class_{k}.jpg")
        img = f'<img src="data:image/jpeg;base64,{b64}"/>' if b64 else ""
        cards.append(
            f'<div class="{cls}">{img}<div class="cn">{name}</div>'
            f'<div class="ct">{" · ".join(traits)}</div></div>'
        )
    st.markdown(f'<div class="gallery">{"".join(cards)}</div>', unsafe_allow_html=True)


def set_values(series):
    for feat, val in series.items():
        st.session_state[f"f_{feat}"] = float(val)


def result_card(pred, nearest):
    pos = min(max(pred, 0), 2) / 2 * 100
    st.markdown(
        f"""
        <div class="result-box">
            <div class="label">Closest class</div>
            <div class="big">{CLASS_NAMES[nearest]}</div>
            <div class="sub">Raw model output: {pred:.3f}</div>
            <div class="scale"><div class="dot" style="left:{pos}%"></div></div>
            <div class="ticks"><span>Class 0</span><span>Class 1</span><span>Class 2</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        st.error(
            "Model or data file not found. Run `python train_model.py` (or your "
            "`notebook.ipynb`) to generate `best_wine_model.pkl` and `wine_data.csv`."
        )
        st.stop()

    model = load_model()
    df = load_data()
    features = [c for c in df.columns if c != "target"]
    class_means = df.groupby("target")[features].mean()

    hero(
        "Which wine is in the glass?",
        "Enter a wine's chemical analysis and the model will tell you which of "
        "the three cultivars it most resembles.",
    )

    for f in features:
        st.session_state.setdefault(f"f_{f}", float(df[f].mean()))

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.subheader("Quick start")
        st.caption("Load the average profile of a class, then adjust from there.")
        for k, name in CLASS_NAMES.items():
            st.button(f"Typical {name}", key=f"qs_{k}", width="stretch",
                      on_click=set_values, args=(class_means.loc[k],))
        st.button("Reset to dataset average", width="stretch",
                  on_click=set_values, args=(df[features].mean(),))
        st.divider()
        st.caption("XYZ Company  \nRegression model trained on the sklearn wine dataset.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        grouped = {f for fs in GROUPS.values() for f in fs}
        layout = dict(GROUPS)
        leftovers = [f for f in features if f not in grouped]
        if leftovers:
            layout["Other"] = leftovers

        for title, feats in layout.items():
            feats = [f for f in feats if f in features]
            if not feats:
                continue
            st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                lo, hi = float(df[feat].min()), float(df[feat].max())
                with cols[i % 2]:
                    unit, desc = MEASURES.get(feat, ("", ""))
                    label = feat.replace("_", " ").capitalize() + (f" ({unit})" if unit else "")
                    dec = 0 if hi - lo > 100 else (1 if hi - lo > 10 else 2)
                    st.slider(label, min_value=lo, max_value=hi, step=(hi - lo) / 100,
                              format=f"%.{dec}f {unit.replace('% vol', 'ABV')}".strip(), key=f"f_{feat}", help=desc or None)

    with right:
        values = {f: st.session_state[f"f_{f}"] for f in features}
        X_new = pd.DataFrame([values], columns=features)
        pred = float(model.predict(X_new)[0])
        nearest = int(min(max(round(pred), 0), 2))

        st.markdown("### Result")
        st.caption("Updates as you move the sliders.")
        result_card(pred, nearest)
        with st.expander("Your measurements"):
            rows = "".join(
                f"<tr><td>{f.replace('_', ' ').capitalize()}</td>"
                f"<td style='text-align:right'><b>{values[f]:.2f}</b> {MEASURES.get(f, ('', ''))[0]}</td></tr>"
                for f in features
            )
            st.markdown(f"<table style='width:100%;font-size:.9rem'>{rows}</table>", unsafe_allow_html=True)
            st.caption("Units are those commonly cited for this dataset; it does not document most of them.")
        class_gallery(nearest, class_means, df[features].mean(), df[features].std())


if __name__ == "__main__":
    main()
