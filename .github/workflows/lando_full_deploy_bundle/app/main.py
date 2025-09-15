import os, json
import torch
from flask import Flask, request, jsonify
from core.trainer import train_with_core
from app.model import LandoNet

MODEL_PATH = os.environ.get('LANDO_MODEL_PATH', 'models/lando_phase2.pt')
VOCAB_PATH = os.environ.get('LANDO_VOCAB_PATH', 'models/vocab.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

def load_vocab(path):
    if not os.path.exists(path):
        return None
    with open(path,'r') as f:
        data = json.load(f)
    return data.get('vocab',[])

vocab = load_vocab(VOCAB_PATH)
model = None
if vocab and os.path.exists(MODEL_PATH):
    try:
        model = LandoNet(len(vocab), base_width=64, grow_factor=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print('Failed to load model:', e)
        model = None

@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({'status':'ok','model_loaded': model is not None})

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = payload.get('message','').strip()
    if not message:
        return jsonify({'error':'no message provided'}),400
    if model is None:
        return jsonify({'error':'model not trained. trigger /core/train or run core/trainer.py locally'}),503
    tokens = set(message.lower().split())
    x = [1.0 if w in tokens else 0.0 for w in vocab]
    xb = torch.tensor([x], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        recon, cscore = model(xb)
    out = recon.cpu().numpy()[0]
    top_idx = out.argsort()[-10:][::-1]
    words = [vocab[i] for i in top_idx if out[i] > 0.0]
    reply = ' '.join(words).strip() or "I don't know yet. Train me with more examples."
    return jsonify({'response': reply, 'contradiction_score': float(cscore.cpu().numpy()[0])})

@app.route('/core/train', methods=['POST'])
def core_train():
    payload = request.get_json(force=True, silent=True) or {}
    epochs = int(payload.get('epochs', 20))
    pairs = payload.get('pairs', None)
    model_obj, vect = train_with_core(pairs=pairs, epochs=epochs, device=DEVICE)
    return jsonify({'status':'trained','epochs':epochs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
