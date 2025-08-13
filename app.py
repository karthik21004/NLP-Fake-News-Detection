
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.caption("TF-IDF + Logistic Regression • SHAP explanations")

@st.cache_resource
def load_assets():
    pipe = joblib.load("model/fake_news_pipeline.joblib")
    background_texts = joblib.load("model/background_texts.joblib")
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    background_X = vectorizer.transform(background_texts)
    explainer = shap.LinearExplainer(clf, background_X, feature_perturbation="interventional")
    return pipe, explainer, vectorizer

pipe, explainer, vectorizer = load_assets()

user_text = st.text_area("Paste your news article here...", height=220)

if st.button("Predict"):
    if user_text.strip():
        proba_true = pipe.predict_proba([user_text])[0, 1]
        pred_label = "Real (True)" if proba_true >= 0.5 else "Fake"
        st.subheader(f"Prediction: **{pred_label}**")
        st.write(f"Confidence (True class): **{proba_true:.3f}**")
        X_u = vectorizer.transform([user_text])
        shap_vals = explainer.shap_values(X_u)
        sv = np.asarray(shap_vals)[0]
        present_idx = X_u.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        tfidf_row = X_u.toarray()[0]
        contrib = pd.DataFrame({
            "term": feature_names[present_idx],
            "shap_value": sv[present_idx],
            "tfidf": tfidf_row[present_idx]
        })
        contrib["impact"] = contrib["shap_value"].abs()
        top = contrib.sort_values("impact", ascending=False).head(20)
        st.markdown("### Top contributing terms")
        st.bar_chart(top.set_index("term")["shap_value"])
        st.dataframe(top[["term", "shap_value", "tfidf"]], use_container_width=True)
    else:
        st.warning("Please enter some text before predicting.")
