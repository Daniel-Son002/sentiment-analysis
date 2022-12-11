from itertools import chain
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data import TextFieldTensors
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training import GradientDescentTrainer
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader

from predictor import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

# Model in AllenNLP represents a model that is trained
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary, positive_label: str = '4') -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features = encoder.get_output_dim(),
                                      out_features = vocab.get_vocab_size('labels'))
        
        positive_index = vocab.get_token_index(positive_label, namespace='labels')
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(self, tokens: TextFieldTensors, label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        
        probs = torch.softmax(logits, dim=-1)
        
        output = {'logits': logits, 'cls_emb': encoder_out, 'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output['loss'] = self.loss_function(logits, label)
            
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset), **self.f1_measure.get_metric(reset)}
    
    
def main():
    reader = StanfordSentimentTreeBankDatasetReader()
    train_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/train.txt'
    dev_path = 'https://s3.amazonaws.com/realworldnlpbook/data/stanfordSentimentTreebank/trees/dev.txt'
    
    sampler = BucketBatchSampler(batch_size = 32, sorting_leys=['tokens'])
    train_data_loader = MultiProcessDataLoader(reader, train_path, batch_sampler=sampler)
    dev_data_loader = MultiProcessDataLoader(reader, dev_path, batch_sampler=sampler)
    
    vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(), dev_data_loader.iter_instances()), min_count={'tokens':3})
    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)
    
    token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFileEmbedder({'tokens': token_embeddings})
    
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    
    model = LstmClassifier(word_embeddings, encoder, vocab)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_data_loader,
                                     validation_data_loader=dev_data_loader,
                                     patience=10,
                                     num_epochs=20,
                                     cuda_device=-1)
    
    trainer.train()
    
    predictor = SentenceClassifierPredictor(model, data_reader=reader)
    logits = predictor.predict('This is the best movie ever!')['logits']
    label_id = np.argmax(logits)
    
    print(model.vocab.get_token_from_index(label_id, 'labels'))
    
if __name__ == '__main__':
    main()
                                    