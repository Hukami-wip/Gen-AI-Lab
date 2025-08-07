import torch
import torch.nn as nn

from .modules.stag_cell import StagCell


class Encoder(nn.Module):
    def __init__(self, input_dim, state_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.cell = StagCell(input_dim, state_dim, num_heads)

    def forward(self, x_seq):
        """
        x_seq: (batch, seq_len, input_dim)
        Returns the final hidden state of the encoder.
        """
        batch_size, seq_len, _ = x_seq.shape
        # The state is (batch, input_dim, state_dim)
        h = torch.zeros(batch_size, self.input_dim, self.state_dim, device=x_seq.device)
        for t in range(seq_len):
            h = self.cell(x_seq[:, t, :], h)
        return h


class Decoder(nn.Module):
    def __init__(self, output_dim, state_dim, input_dim, embedding_dim, num_heads):
        super().__init__()
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.output_dim, self.embedding_dim)
        self.cell = StagCell(embedding_dim, self.state_dim, num_heads)

        # It flattens the state and projects it to a single vector.
        self.readout_proj = nn.Linear(self.input_dim * self.state_dim, self.state_dim)

        # The final layer maps the state to logits over the possible output indices
        self.fc = nn.Linear(self.state_dim, self.output_dim)

    def forward(self, x_t, h_prev):
        """
        x_t: The previous token (either from target or last prediction) -> (batch, 1)
        h_prev: The previous hidden state from the decoder -> (batch, input_dim, state_dim)
        """
        # Embed the input token index
        x_embedded = self.embedding(x_t.squeeze(1))

        h_next = self.cell(x_embedded, h_prev)
        # Project state to logits
        batch_size = h_next.shape[0]
        flattened_state = h_next.view(batch_size, -1)
        readout_vector = self.readout_proj(flattened_state)

        # Project the learned readout vector to the final output logits
        output_logits = self.fc(readout_vector)

        return output_logits, h_next


class StagSorter(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, embedding_dim, num_heads):
        super().__init__()
        self.encoder = Encoder(input_dim, state_dim, num_heads)
        self.decoder = Decoder(
            output_dim, state_dim, input_dim, embedding_dim, num_heads
        )

    def forward(self, x_input):
        batch_size, seq_len, _ = x_input.shape

        # --- Encoder Pass ---
        encoder_hidden = self.encoder(x_input)

        # --- Decoder Pass ---
        decoder_hidden = encoder_hidden
        outputs = []

        # Start token is the index 0
        decoder_input = torch.zeros(
            batch_size, 1, dtype=torch.long, device=x_input.device
        )

        for t in range(seq_len):
            decoder_output_logits, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden
            )
            outputs.append(decoder_output_logits)

            # Get the predicted class index for the next step
            top1 = decoder_output_logits.argmax(1).unsqueeze(1)
            decoder_input = top1

        return torch.stack(outputs, dim=1)
