# Rice Classification Model

A portfolio-ready **Machine Learning + Streamlit + Python** project for rice-grain image classification.

## Supported classes
- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

## Features
- Upload rice images
- Image preview
- Rice variety prediction
- Confidence score
- Per-class probability bars
- Dark portfolio UI
- Optional trained scikit-learn model
- Demo fallback when no model is present

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Train your own model

Create:

```text
dataset/
  Arborio/
  Basmati/
  Ipsala/
  Jasmine/
  Karacadag/
```

Put training images inside each folder and run:

```bash
python train_model.py
```

The trained model will be saved as:

```text
model/rice_classifier.pkl
```

## Important
Without a trained model file, the app uses demo predictions for interface/portfolio presentation only.
