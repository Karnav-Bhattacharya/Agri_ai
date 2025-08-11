import fasttext

# Download model once:
# !wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

#? I am going to change fasttext library for this venv line 228
# return labels, np.array(probs, copy=False)
# to
# return labels, np.asarray(probs)


model = fasttext.load_model("lid.176.bin")

def detect_language_with_confidence_fasttext(text, threshold=0.85):
    predictions = model.predict(text, k=1)
    lang_code = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]

    if confidence < threshold:
        return None, confidence
    return lang_code, confidence

# Example usage:
query = "আজ আকাশটা খুব সুন্দর, হালকা মেঘ আর ঠান্ডা হাওয়া বইছে।"
lang, conf = detect_language_with_confidence_fasttext(query)

if lang is None:
    print(f"⚠️ Low confidence ({conf:.2f}) — please clarify.")
else:
    print(f"✅ Detected: {lang} (confidence {conf:.2f})")