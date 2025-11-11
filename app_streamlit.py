# app_streamlit.py
import streamlit as st
import os
import joblib
import torch
import numpy as np

# --- Optionnel : si tu veux utiliser BERT, installe transformers
# pip install streamlit joblib transformers torch

st.set_page_config(page_title="BBC Document Classifier", layout="centered")

st.title("Document Classifier — BBC ")
st.write("Entrez une phrase ou un court paragraphe. L'application prédit la catégorie.")

# Choix du modèle
use_bert = st.checkbox("Utiliser BERT ", value=False)

# Paths attendus (modifie si nécessaire)
PIPELINE_PATH = "tfidf_svm_bbc.pkl"    # pipeline sklearn: TF-IDF + SVM (joblib)
BERT_DIR = "bert_bbc_out"              # dossier contenant model + tokenizer (Trainer.save_model)

# Charger modèle TF-IDF+SVM si disponible
sk_pipeline = None
if os.path.exists(PIPELINE_PATH):
    try:
        sk_pipeline = joblib.load(PIPELINE_PATH)
    except Exception as e:
        st.warning(f"Impossible de charger {PIPELINE_PATH}: {e}")
else:
    st.info(f"Pipeline TF-IDF+SVM non trouvé ({PIPELINE_PATH}). Sauvegarde-le comme indiqué dans le notebook.")

# Charger modèle BERT si demandé et disponible
tokenizer = None
bert_model = None
if use_bert:
    if os.path.isdir(BERT_DIR):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(BERT_DIR, local_files_only=True)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR, local_files_only=True)
            bert_model.eval()
        except Exception as e:
            st.error(f"Echec chargement BERT depuis '{BERT_DIR}': {e}")
            tokenizer = None
            bert_model = None
    else:
        st.error(f"Directory BERT non trouvé: {BERT_DIR}")

# Input utilisateur
user_text = st.text_area("Texte à classifier", height=150)

if st.button("Classify"):

    if not user_text or user_text.strip() == "":
        st.warning("Merci d'entrer une phrase ou un paragraphe.")
    else:
        if use_bert:
            if bert_model is None or tokenizer is None:
                st.error("BERT non disponible. Décoche la case BERT ou fournis le dossier 'bert_bbc_out'.")
            else:
                # Inference BERT (CPU)
                inputs = tokenizer(user_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    logits = outputs.logits.cpu().numpy()
                    pred_id = int(np.argmax(logits, axis=-1)[0])
                # essayer de récupérer id2label (si sauvegardé)
                id2label = None
                if hasattr(bert_model.config, "id2label"):
                    id2label = bert_model.config.id2label
                if id2label:
                    label = id2label.get(str(pred_id), id2label.get(pred_id, str(pred_id)))
                else:
                    label = str(pred_id)
                st.success(f"✅ Predicted label: **{label}**")
        else:
            # Mode TF-IDF + SVM
            if sk_pipeline is None:
                st.error("Pipeline TF-IDF+SVM non trouvé. Sauvegarde-le d'abord (voir instructions ci-dessous).")
            else:
                pred = sk_pipeline.predict([user_text])[0]
                # si ton pipeline rend des ids, convertis ici ; sinon, affiche direct
                st.success(f"✅ Predicted label: **{pred}**")

st.markdown("---")
