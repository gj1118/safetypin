# Custom Good/Bad Classifier

Custom-trained classifiers for content moderation (good/bad classification).

## What This Is

- **Image Classifier**: ResNet18 CNN trained on your images
- **Text Classifier**: LSTM neural network trained on chat messages

Both classifiers determine if content is appropriate ("good") or inappropriate ("bad").

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python local_server.py
```

The server runs at `http://localhost:8000`

## API Endpoints

### Classify Image
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_encoded_image>"}'
```

### Classify Text
```bash
curl -X POST http://localhost:8000/classify_text \
  -H "Content-Type: application/json" \
  -d '{"text": "your message here"}'
```

## Project Structure

```
.
├── local_server.py           # Main server (runs classifier)
├── train_classifier.py       # Training script (for retraining)
├── requirements.txt          # Dependencies
├── data/
│   ├── train.json           # Text training data
│   └── images/             # Image training data
│       ├── bad/            # Bad example images
│       └── good/           # Good example images
└── output/
    ├── image_classifier.pth    # Trained image model
    └── text_model_weights.pth  # Trained text model
```

## Retraining

To improve accuracy, retrain with more data:

```bash
python train_classifier.py
```

This will:
- Retrain image classifier on images in `data/images/`
- Retrain text classifier on chat messages in `data/train.json`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Flask

Install with: `pip install -r requirements.txt`

## Notes

- Models run on CPU (works on MacBook M1/M2/M3)
- No external API calls - fully local
- Training data: 347 images, 5000 text samples
