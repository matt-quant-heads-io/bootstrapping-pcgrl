import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNStateEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # CNN architecture for 1056x1056x3 RGB images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Calculate the size of flattened features
        self.flat_size = 512 * (1056 // (2**5)) * (1056 // (2**5))
        
        # Final fully connected layer to match hidden dimension
        self.fc = nn.Linear(self.flat_size, hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 1056, 1056)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten the features
        x = x.view(-1, self.flat_size)
        
        # Project to hidden dimension
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1024):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim=1056*1056*3,  # RGB image dimensions
        action_dim=8,
        hidden_dim=128,
        max_seq_length=20,
        max_ep_len=1000,
        n_layer=3,
        n_head=4,
        n_inner=256,
        activation_function='relu',
        n_positions=1024,
        dropout=0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        # State encoder
        self.state_encoder = CNNStateEncoder(hidden_dim)
        
        # Action and reward embeddings
        self.action_embeddings = nn.Linear(action_dim, hidden_dim)
        self.reward_embeddings = nn.Linear(1, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=n_inner,
            dropout=dropout,
            activation=activation_function
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layer)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assuming actions are normalized to [-1, 1]
        )
        
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Embed states using CNN
        state_embeddings = torch.stack([self.state_encoder(states[:, i]) 
                                      for i in range(seq_length)], dim=1)
        
        # Embed actions, rewards, and returns
        action_embeddings = self.action_embeddings(actions)
        reward_embeddings = self.reward_embeddings(rewards.unsqueeze(-1))
        returns_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))

        # Time embeddings
        time_embeddings = self.embed_timestep(timesteps)

        # Compose the sequence
        sequence = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings),
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_dim)
        
        # Add positional encodings
        sequence = self.pos_encoder(sequence)
        
        # Create attention mask that allows each token to attend only to previous tokens
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through transformer
        sequence_output = self.transformer(sequence.transpose(0, 1),
                                        src_key_padding_mask=~attention_mask.bool()).transpose(0, 1)
        
        # Get actions output
        action_preds = sequence_output[:, 1::3]  # Only use state embeddings to predict actions
        action_preds = self.action_head(action_preds)
        
        return action_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # Model inference for a single sequence
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        rewards = rewards.unsqueeze(0)
        returns_to_go = returns_to_go.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        
        action_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, **kwargs
        )
        
        return action_preds[0, -1]  # Return last predicted action

# Example usage
def initialize_decision_transformer():
    model = DecisionTransformer(
        state_dim=1056*1056*3,
        action_dim=8,
        hidden_dim=128,
        max_seq_length=20,
        max_ep_len=1000,
        n_layer=3
    )
    return model

# Training loop example
def train_step(model, optimizer, batch):
    states, actions, rewards, returns_to_go, timesteps, attention_mask = batch
    
    action_preds = model.forward(
        states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask
    )
    
    loss = F.mse_loss(action_preds, actions, reduction='mean')
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    
    return loss.item()

# Example of creating the model and optimizer
def setup_training():
    model = initialize_decision_transformer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    return model, optimizer