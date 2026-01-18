import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_tokenizer(path='tokenizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path='cnn_sent_model.h5'):
    """Load a Keras model. Try normal load first; on optimizer/kwarg errors fall back to compile=False.

    Some saved models include optimizer configuration that may be incompatible with the current
    TensorFlow/Keras version (e.g., `weight_decay` in optimizer kwargs). In that case we load with
    `compile=False` which is sufficient for inference.
    """
    try:
        return load_model(path)
    except Exception as e:
        # Attempt to load without compiling (ignore optimizer state) for inference
        try:
            model = load_model(path, compile=False)
            # Mark so the caller can warn the user if desired
            setattr(model, '__loaded_without_compile__', True)
            return model
        except Exception:
            # Re-raise original exception to preserve the original message
            raise e


def get_maxlen_from_model(model):
    # Try common ways to get expected input length
    try:
        # model.input_shape is (None, maxlen) for sequential models
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and shape[1] is not None:
            return int(shape[1])
    except Exception:
        pass
    try:
        # Embedding layer often has input_length attribute
        for layer in model.layers:
            if hasattr(layer, 'input_length') and layer.input_length is not None:
                return int(layer.input_length)
    except Exception:
        pass
    # fallback
    return 100


def predict_text(text, tokenizer, model, maxlen):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, padding='post', maxlen=maxlen)
    preds = model.predict(padded)
    # handle different output shapes
    if isinstance(preds, np.ndarray):
        pred = preds[0]
        # binary sigmoid output (shape (1,) or (1,1))
        if pred.size == 1:
            prob = float(pred.ravel()[0])
            label = 'Positive' if prob >= 0.5 else 'Negative'
            return label, prob
        # multiclass softmax (e.g., [neg_prob, pos_prob])
        else:
            class_idx = int(np.argmax(pred))
            prob = float(pred[class_idx])
            label = 'Positive' if class_idx == 1 else 'Negative'
            return label, prob
    # fallback
    return 'Unknown', 0.0


def main():
    st.set_page_config(page_title='Text Sentiment Classifier', layout='centered')
    st.title('📄 CNN Text Sentiment Classifier')

    # Load artifacts
    with st.spinner('Loading model and tokenizer...'):
        try:
            tokenizer = load_tokenizer('tokenizer.pkl')
        except FileNotFoundError:
            st.error('`tokenizer.pkl` not found in the working directory.')
            return
        try:
            model = load_keras_model('cnn_sent_model.h5')
        except Exception as e:
            st.error(f'Error loading model: {e}')
            return
        maxlen = get_maxlen_from_model(model)

    st.markdown(f'**Model input maxlen:** `{maxlen}`')

    inp = st.text_area('Enter text to classify', height=150)
    if st.button('Classify'):
        if not inp.strip():
            st.warning('Please enter some text.')
        else:
            label, prob = predict_text(inp, tokenizer, model, maxlen)
            st.subheader(f'Result: **{label}**')
            st.write(f'Confidence: **{prob:.3f}**')
            st.progress(min(max(prob, 0.0), 1.0))

    st.markdown('---')

    st.caption('Built with Streamlit — run with `streamlit run app.py`')


if __name__ == '__main__':
    main()
