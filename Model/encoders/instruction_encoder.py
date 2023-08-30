import gzip
import json

import torch
import torch.nn as nn

from Model.utils.CN import CN as basic_CN
from src.common.param import args


class InstructionEncoder(nn.Module):
    #
    def __init__(self):
        r"""An encoder that uses RNN to encode an instruction. Returns
        the final hidden state after processing the instruction sequence.

        Args:
            config: must have
                embedding_size: The dimension of each embedding vector
                hidden_size: The hidden (output) size
                rnn_type: The RNN cell type.  Must be GRU or LSTM
                final_state_only: Whether or not to return just the final state
        """
        super().__init__()

        CN = basic_CN.clone()

        if args.policy_type in ['seq2seq', 'unet', 'vlnbert']:
            CN.rnn_type = 'LSTM'
            CN.embedding_size = 50
            CN.hidden_size = 128
            CN.bidirectional = False
            CN.use_pretrained_embeddings = False
            CN.embedding_file = None
            CN.fine_tune_embeddings = False
            CN.vocab_size = args.vocab_size
            CN.final_state_only = True
        elif args.policy_type == 'cma':
            CN.rnn_type = 'LSTM'
            CN.embedding_size = 50
            CN.hidden_size = 128
            CN.bidirectional = True
            CN.use_pretrained_embeddings = False
            CN.embedding_file = None
            CN.fine_tune_embeddings = False
            CN.vocab_size = args.vocab_size
            CN.final_state_only = False
        else:
            raise NotImplementedError

        config = CN.clone()
        self.config = config

        self.encoder_rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        if self.config.use_pretrained_embeddings:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=self._load_embeddings(),
                freeze=not self.config.fine_tune_embeddings,
            )
        else:  # each embedding initialized to sampled Gaussian
            self.embedding_layer = nn.Embedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.embedding_size,
                padding_idx=0,
            )

    #
    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    #
    def _load_embeddings(self):
        """Loads word embeddings from a pretrained embeddings file.
        PAD: index 0. [0.0, ... 0.0]
        UNK: index 1. mean of all R2R word embeddings: [mean_0, ..., mean_n]
        why UNK is averaged: https://bit.ly/3u3hkYg
        Returns:
            embeddings tensor of size [num_words x embedding_dim]
        """
        with gzip.open(self.config.embedding_file, "rt") as f:
            embeddings = torch.tensor(json.load(f))
        return embeddings

    #
    def forward(self, observations):
        """
        Tensor sizes after computation:
            instruction: [batch_size x seq_length]
            lengths: [batch_size]
            hidden_state: [batch_size x hidden_size]
        """
        instruction = observations["instruction"].long()
        lengths = (instruction != 0.0).long().sum(dim=1)
        instruction = self.embedding_layer(instruction)

        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            instruction, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[
                0
            ].permute(0, 2, 1)


class InstructionBertEncoder(nn.Module):
    #
    def __init__(self):
        super().__init__()

        CN = basic_CN.clone()

        if args.policy_type in ['seq2seq', 'unet', 'vlnbert']:
            CN.rnn_type = 'LSTM'
            CN.embedding_size = 50
            CN.hidden_size = 128
            CN.bidirectional = False
            CN.use_pretrained_embeddings = False
            CN.embedding_file = None
            CN.fine_tune_embeddings = False
            CN.vocab_size = args.vocab_size
            CN.final_state_only = True
        elif args.policy_type == 'cma':
            CN.rnn_type = 'LSTM'
            CN.embedding_size = 50
            CN.hidden_size = 128
            CN.bidirectional = True
            CN.use_pretrained_embeddings = False
            CN.embedding_file = None
            CN.fine_tune_embeddings = False
            CN.vocab_size = args.vocab_size
            CN.final_state_only = False
        elif args.policy_type == 'hcm':
            CN.rnn_type = 'LSTM'
            CN.embedding_size = 768
            CN.hidden_size = 256
            CN.bidirectional = False
            CN.use_pretrained_embeddings = False
            CN.embedding_file = None
            CN.fine_tune_embeddings = False
            CN.vocab_size = args.vocab_size
            CN.final_state_only = False
        else:
            raise NotImplementedError

        config = CN.clone()
        self.config = config

        self.encoder_rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            bidirectional=config.bidirectional,
        )

        from transformers import BertModel
        self.embedding_layer = BertModel.from_pretrained("bert-base-uncased")
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        self.embedding_layer.eval()

        self.drop = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(
                self.embedding_layer.config.hidden_size,
                config.embedding_size,
            ),
            nn.Tanh(),
        )

    @property
    def output_size(self):
        return self.config.hidden_size * (1 + int(self.config.bidirectional))

    def forward(self, observations):
        instruction = observations["instruction"].long()
        lengths = (instruction != 0.0).long().sum(dim=1)

        self.embedding_layer.eval()
        with torch.no_grad():
            embedded = self.embedding_layer(instruction)
            embedded = embedded[0]

        # embedded = self.drop(embedded)

        embedded = self.fc(embedded)

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, final_state = self.encoder_rnn(packed_seq)

        final_state = final_state[0]

        if self.config.final_state_only:
            return final_state.squeeze(0)
        else:
            return nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0].permute(0, 2, 1)

