import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

# Define the custom model architecture (copied from train_text.py)
class DistilBertWithReasons(nn.Module):
    def __init__(self, num_labels=2, num_reasons=25):
        super().__init__()
        self.num_labels = num_labels
        self.num_reasons = num_reasons

        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.6)

        # Classification head (good/bad) with additional layer
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels)
        )

        # Reason prediction head with additional layer
        self.reason_classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_reasons)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)

        # Classification logits
        classification_logits = self.classifier(pooled_output)

        # Reason logits
        reason_logits = self.reason_classifier(pooled_output)

        # Return tuple for ONNX export
        return classification_logits, reason_logits

# Load the model
model = DistilBertWithReasons(num_labels=2, num_reasons=25)
model.load_state_dict(torch.load('output/bert_text_with_reasons/model.pth', map_location='cpu'))
model.eval()

dummy_input_ids = torch.randint(0, 30000, (1, 128))
dummy_attention_mask = torch.ones(1, 128)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    'output/text_classifier.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['classification_logits', 'reason_logits'],
    dynamic_axes={
        'input_ids': {1: 'seq_len'},
        'attention_mask': {1: 'seq_len'}
    },
    opset_version=17
)
print('Text classifier exported to output/text_classifier.onnx')
