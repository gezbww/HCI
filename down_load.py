# from transformers import HubertModel

# model = HubertModel.from_pretrained("facebook/hubert-xlarge-ls960-ft")
# model.save_pretrained("pretrained_models/hubert-xlarge-ls960-ft")
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
processor.save_pretrained("pretrained_models/hubert-xlarge-ls960-ft")

model = Wav2Vec2Model.from_pretrained("facebook/hubert-xlarge-ls960-ft")
model.save_pretrained("pretrained_models/hubert-xlarge-ls960-ft")
