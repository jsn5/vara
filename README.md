# Vara - Malayalam Handwriting Generation

A Conditional Sketch-RNN model for generating Malayalam handwriting from text input. Uses a VAE architecture with Mixture Density Network (MDN) output for stroke generation.

## Features

- **Data Collection**: Web-based interface for collecting handwriting samples with Wacom tablet support
- **Conditional Sketch-RNN**: Text-conditioned handwriting generation using character-level encoding
- **Web Demo**: Browser-based inference using ONNX Runtime Web
- **Dynamic Corpus**: Fetches Malayalam words from Wikipedia for training variety

## Architecture

- **Text Encoder**: Bidirectional LSTM for Malayalam character sequences
- **Stroke Encoder**: Bidirectional LSTM VAE for encoding handwriting strokes
- **Decoder**: Autoregressive LSTM with MDN output (20 mixture components)
- **Format**: Stroke-5 format `[dx, dy, p1, p2, p3]` where p1=pen down, p2=pen up, p3=end

## Installation

```bash
# Clone the repository
git clone https://github.com/jsn5/vara.git
cd vara

# Install dependencies
pip install torch flask numpy matplotlib requests onnx onnxruntime
```

## Usage

### 1. Data Collection

Start the web app for collecting handwriting samples:

```bash
python app.py
```

Open http://localhost:5000 and draw Malayalam characters/words using a stylus or mouse.

### 2. Training

Train the Conditional Sketch-RNN model:

```bash
# Consolidate collected data
python -c "from train import *; # data consolidation"

# Train the model
python train.py --epochs 100 --batch_size 32
```

### 3. Generation

Generate handwriting from text:

```python
from models.sketch_rnn import ConditionalSketchRNN
import torch

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
model = ConditionalSketchRNN(vocab_size=156, ...)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate
text_ids = encode_text("മലയാളം")  # Malayalam text
strokes = model.sample(text_ids, temperature=0.4)
```

### 4. Web Demo

Export to ONNX and run the web demo:

```bash
# Export model to ONNX
python export_onnx.py

# Start server
python -m http.server 8080

# Open http://localhost:8080/web_demo/
```

## Project Structure

```
vara/
├── app.py                 # Flask data collection server
├── train.py               # Training script
├── export_onnx.py         # ONNX export for web inference
├── corpus_loader.py       # Wikipedia corpus fetcher
├── models/
│   ├── sketch_rnn.py      # Conditional Sketch-RNN model
│   └── dataset.py         # Dataset and data loading
├── web_demo/
│   ├── index.html         # Web demo interface
│   └── app.js             # ONNX inference in browser
├── data/                  # Collected handwriting data
├── checkpoints/           # Trained model checkpoints
└── onnx_models/           # Exported ONNX models
```

## Model Details

- **Vocabulary**: 156 tokens (Malayalam Unicode range U+0D00-U+0D7F + special tokens)
- **Text Embedding**: 128-dim embeddings → 256-dim bidirectional LSTM → 512-dim encoding
- **Latent Space**: 128-dim VAE latent vector
- **Decoder**: 1024-dim LSTM with 20-component MDN output
- **Total Parameters**: ~11.5M

## Limitations

- Text conditioning learns general patterns but not precise character-to-stroke alignment
- Best results with single characters; longer words may not terminate properly
- Would benefit from attention mechanisms for better text-stroke alignment

## References

- [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477) - Ha & Eck, 2017
- [Teaching Machines to Draw](https://research.google/blog/teaching-machines-to-draw/) - Google AI Blog

## License

MIT License
