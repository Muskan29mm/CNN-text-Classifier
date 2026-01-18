# Streamlit Text Sentiment Classifier

This app loads the trained `cnn_sent_model.h5` and `tokenizer.pkl` from the repository and provides a Streamlit UI to classify single texts or batch CSV files.

## Quick start

1. (Optional) Create a virtual environment and activate it

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run app.py
```

Note: make sure to type the full command `streamlit run app.py` (typing `strea` or other partial commands will raise "command not found").

4. Enter text into the box and click **Classify**.

If your model or tokenizer filenames differ, update the references in `app.py` accordingly.

Note about model loading errors:
- If you see an error like: `weight_decay is not a valid argument, kwargs should be empty for optimizer_experimental.Optimize` when the app tries to load the model, it means the saved model includes optimizer metadata incompatible with your current TensorFlow/Keras. To fix: open a Python session, load the model with `compile=False` and re-save without optimizer state:

```python
from tensorflow.keras.models import load_model
m = load_model('cnn_sent_model.h5', compile=False)
m.save('cnn_sent_model.h5', include_optimizer=False)
```

Then restart the Streamlit app which will load the model normally.