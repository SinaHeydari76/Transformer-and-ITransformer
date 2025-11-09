import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
# %%
# Path to the exchange rate data file
data_path = 'exchange_rate.txt'

# Load the exchange rate data into a pandas DataFrame
df = pd.read_csv(data_path)

# Assign column names to the DataFrame
column_names = ['Australia',
                'British',
                'Canada',
                'Switzerland',
                'China',
                'Japan',
                'New Zealand',
                'Singapore']
df.columns = column_names

# Display the first few rows of the DataFrame
df.head()
# %%
# Plotting the exchange rates over time
plt.figure(figsize=(12, 6))
for currency in df.columns:
    plt.plot(df.index,
             df[currency],
             label=currency)
    
plt.title('Exchange Rates Over Time')
plt.xlabel('Time')
plt.ylabel('Exchange Rate')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the distribution of exchange rates for each currency
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Distribution of Exchange Rates for Each Currency')
plt.ylabel('Exchange Rate')
plt.xticks(rotation=45)
plt.show()

# Print the number of rows and columns in the DataFrame
num_rows, num_columns = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)
# %%
# Custom Dataset class for the exchange rate data
class ExchangeRateDataset(Dataset):
    def __init__(self,
                 data,
                 input_length,
                 output_length):
        """
        Initialize the dataset.
        :param data: The data array.
        :param input_length: Length of the input sequence.
        :param output_length: Length of the output sequence.
        """
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data) \
            - (self.input_length + \
               self.output_length) + 1

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        :param index: Index of the sample.
        :return: Tuple of input sequence and output sequence.
        """
        index_end = index + self.input_length
        input_seq = self.data[index:index_end]
        output_seq = self.data[index_end:index_end + \
                               self.output_length]
        
        return input_seq, output_seq

# Parameters for the DataLoader
input_dimension = 10  # Lookback time series length
output_dimension = 1  # Predicted length
batch_size = 64
data = df.values

# Split the data into training and test sets
train_data, test_data = train_test_split(data,
                                         test_size=0.2,
                                         shuffle=False)

# Create DataLoader for training data
train_dataset = ExchangeRateDataset(train_data,
                                    input_dimension,
                                    output_dimension)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# Create DataLoader for test data
test_dataset = ExchangeRateDataset(test_data,
                                   input_dimension,
                                   output_dimension)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)
# %%
# Custom Embedding class for the transformer model
class Embedding(nn.Module):
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 intermediate_dimension=1024):
        """
        Initialize the embedding layer.
        :param input_dimension: Dimension of the input.
        :param output_dimension: Dimension of the output.
        :param intermediate_dimension: Dimension of the intermediate layer.
        """
        super(Embedding, self).__init__()
        
        self.fullyconnected = nn.Sequential(
            nn.Linear(input_dimension,
                      intermediate_dimension),
            nn.ReLU(),
            nn.Linear(intermediate_dimension,
                      output_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the embedding layer.
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.fullyconnected(x)

# Positional Embedding class to add positional information to the embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self,
                 dimension,
                 max_length=1000):
        """
        Initialize the positional embedding.
        :param dimension: Dimension of the embeddings.
        :param max_length: Maximum length of the sequences.
        """
        super(PositionalEmbedding, self).__init__()
        
        positionalembedding = torch.zeros(max_length,
                                          dimension).float()
        positionalembedding.requires_grad = False
        
        position = torch.arange(0, max_length).float().unsqueeze(1)
        
        division_term = (torch.arange(0,
                                 dimension,
                                 2).float() * \
                    -(math.log(10000.0) / dimension)).exp()
        
        positionalembedding[:, 0::2] = torch.sin(position * \
                                                 division_term)
        positionalembedding[:, 1::2] = torch.cos(position * \
                                                 division_term)
            
        positionalembedding = positionalembedding.unsqueeze(0)
        
        self.register_buffer('positionalembedding',
                             positionalembedding)

    def forward(self, x):
        """
        Forward pass to add positional information to the input tensor.
        :param x: Input tensor.
        :return: Tensor with added positional information.
        """
        return self.positionalembedding[:, :x.size(1)]

# 

# Encoder Block class for the transformer model
class EncoderBlock(nn.Module):
    def __init__(self,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the encoder block.
        :param dimension: Dimension of the input.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(EncoderBlock, self).__init__()
        
        self.Attention = nn.MultiheadAttention(dimension,
                                               heads,
                                               dropout=dropout,
                                               batch_first=True)
        
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        
        self.fullyconnected = nn.Sequential(
            nn.Linear(dimension,
                      intermediate_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dimension,
                      dimension),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass through the encoder block.
        :param x: Input tensor.
        :return: Output tensor.
        """
        attention_out, attention_weights = self.Attention(x,
                                                          x,
                                                          x)
        linear_out = self.fullyconnected(x)

        x = self.norm1(x + attention_out)
        x = self.norm2(x + linear_out)

        return x

# Encoder class composed of multiple Encoder Blocks
class Encoder(nn.Module):
    def __init__(self,
                 N,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the encoder.
        :param N: Number of encoder blocks.
        :param dimension: Dimension of the input.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(Encoder, self).__init__()
        
        self.encoderblocks = nn.ModuleList(
                          [EncoderBlock(dimension,
                          heads,
                          intermediate_dimension,
                          dropout) for _ in range(N)]
                          )

    def forward(self, x):
        """
        Forward pass through the encoder.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for block in self.encoderblocks:
            x = block(x)
        return x
# 
# Decoder Block class for the transformer model
class DecoderBlock(nn.Module):
    def __init__(self,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the decoder block.
        :param dimension: Dimension of the input.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(DecoderBlock, self).__init__()
        
        self.selfAttention = nn.MultiheadAttention(dimension,
                                                   heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)
        
        self.crossAttention = nn.MultiheadAttention(dimension,
                                                    heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        
        self.fullyconnected = nn.Sequential(
            nn.Linear(dimension, intermediate_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dimension, dimension),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self,
                x,
                encoder_out):
        """
        Forward pass through the decoder block.
        :param x: Input tensor.
        :param enc_out: Encoder output tensor.
        :return: Output tensor.
        """
        mask = torch.tril(torch.ones(x.shape[1],
                                     x.shape[1])).to(self.device)
        
        attention_out, attention_weights = self.selfAttention(x,
                                                  x,
                                                  x,
                                                  attn_mask=mask)
        
        x = self.norm1(attention_out + x)
        
        attention_out, attention_weights = self.crossAttention(x,
                                                   encoder_out,
                                                   encoder_out)
        x = self.norm2(x + attention_out)
        fullyconnected_out = self.fullyconnected(x)
        x = self.norm3(fullyconnected_out + x)
        return x

# Decoder class composed of multiple Decoder Blocks
class Decoder(nn.Module):
    def __init__(self,
                 N,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the decoder.
        :param N: Number of decoder blocks.
        :param dimension: Dimension of the input.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dimension,
                          heads,
                          intermediate_dimension,
                          dropout) for _ in range(N)])

    def forward(self, x, enc_out):
        """
        Forward pass through the decoder.
        :param x: Input tensor.
        :param enc_out: Encoder output tensor.
        :return: Output tensor.
        """
        for block in self.decoder_blocks:
            x = block(x, enc_out)
        return x

# Transformer model class combining the encoder and decoder
class Transformer(nn.Module):
    def __init__(self,
                 N,
                 input_dimension,
                 output_dimension,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the transformer model.
        :param N: Number of encoder and decoder blocks.
        :param input_dimension: Dimension of the input.
        :param output_dimension: Dimension of the output.
        :param dimension: Dimension of the embeddings.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(Transformer, self).__init__()
        self.embedder = Embedding(input_dimension,
                                  dimension)
        self.positional_embedder = PositionalEmbedding(dimension)
        
        self.encoder = Encoder(N,
                               dimension,
                               heads,
                               intermediate_dimension,
                               dropout)
        self.decoder = Decoder(N,
                               dimension,
                               heads,
                               intermediate_dimension,
                               dropout)
        
        self.linear = nn.Sequential(
            nn.Linear(dimension, intermediate_dimension),
            nn.ReLU(),
            nn.Linear(intermediate_dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, output_dimension)
        )

    def forward(self, x_input, y_output):
        """
        Forward pass through the transformer model.
        :param x_input: Input tensor.
        :param y_output: Output tensor.
        :return: Output tensor.
        """
        x_input = self.embedder(x_input) + \
            self.positional_embedder(x_input)
        
        y_output = self.embedder(y_output) + \
            self.positional_embedder(y_output)
        
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(y_output,
                                      encoder_output)
        
        out = self.linear(decoder_output)
        return out
# %% 
# iTransformer
class iTransformer(nn.Module):
    def __init__(self,
                 N,
                 input_dimension,
                 output_dimension,
                 dimension,
                 heads,
                 intermediate_dimension,
                 dropout=0.1):
        """
        Initialize the transformer model.
        :param N: Number of encoder and decoder blocks.
        :param input_dimension: Dimension of the input.
        :param output_dimension: Dimension of the output.
        :param dimension: Dimension of the embeddings.
        :param heads: Number of attention heads.
        :param intermediate_dimension: Dimension of the intermediate layer.
        :param dropout: Dropout rate.
        """
        super(iTransformer, self).__init__()
        
        self.embedder = Embedding(input_dimension,
                                  dimension)

        self.encoder = Encoder(N, dimension,
                               heads,
                               intermediate_dimension,
                               dropout)

        self.projection = nn.Sequential(
            nn.Linear(dimension,
                      intermediate_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dimension,
                      output_dimension),
            )
        
    def forward(self,x):
        
        """
        Forward pass through the itransformer model.
        :param x: Input tensor.
        :return: Output tensor.
        """
        
        x = x.permute(0,2,1)
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.projection(x)
        x = x.permute(0,2,1)
        
        return x
# %%
# Check for device availability and,
# create model,
# loss function and,
# optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformer_model = Transformer(N=6,
                                input_dimension=8,
                                output_dimension=8,
                                dimension=512,
                                heads=4,
                                intermediate_dimension=1024).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=2e-4)
# %%
# Training the transformer model
epochs = 100
losses = []

for epoch in range(epochs):
    running_loss = 0
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        # Assuming the -1 as the starter token
        decoder_in = (torch.ones_like(y) * -1).float().to(device)
        y_pred = transformer_model(x, decoder_in)
        loss = criterion(y_pred, y)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print(f'''Epoch {epoch+1}/{epochs} 
          --> Mean Saquared Eror Loss --> {running_loss}''')
    losses.append(running_loss)
# %%
# Plotting the training loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error loss')
plt.show()

# Initialize tensors to store predictions and true values
y_preds = torch.tensor([]).to(device)
y_true = torch.tensor([])

# Evaluation on the test set
with torch.no_grad():
    for x, y in test_loader:
        x = x.float().to(device)
        decoder_input = (torch.ones_like(y) * -1).float().to(device)
        y_pred = transformer_model(x, decoder_input)
        y_preds = torch.concat([y_preds, y_pred], dim=0)
        y_true = torch.concat([y_true, y], dim=0)

# Save the trained model
torch.save(transformer_model.state_dict(), "transformer_model.pt")

# Convert predictions and true values to numpy arrays
y_preds = y_preds.cpu().numpy()
y_true = y_true.numpy()
# %%
# Calculate and print,
# the Mean Squared Error (MSE) and,
# Mean Absolute Error (MAE)
mse_transformer = mean_squared_error(y_true.squeeze(),
                                     y_preds.squeeze())
mae_transformer = mean_absolute_error(y_true.squeeze(),
                                      y_preds.squeeze())
print(f'''Mean Squared Eror: {mse_transformer} \n
Mean Absolute Eror: {mae_transformer}''')
# %% # Training the itransformer model
iTransformer_model = iTransformer(6,
                                  10,
                                  1,
                                  512,
                                  4,
                                  1024).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(iTransformer_model.parameters(), lr=2e-4)
# %%
epochs= 100
losses = []
for epoch in range(epochs):
  running_loss = 0
  for x, y in train_loader:
      x = x.float().to(device)
      y = y.float().to(device)

      optimizer.zero_grad()

      y_pred = iTransformer_model(x)

      loss = criterion(y_pred, y)
      loss.backward()

      running_loss += loss.item()
      optimizer.step()
  print(f'''Epoch {epoch+1}/{epochs} 
       --> Mean Saquared Eror Loss --> {running_loss}''')
  losses.append(running_loss)
# %%
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Eror loss')
plt.show()
# %%
y_preds = torch.tensor([]).to(device)
y_true = torch.tensor([])
with torch.no_grad():
  for x,y in test_loader:
    x = x.float().to(device)
    y_pred = iTransformer_model(x)
    y_preds = torch.concat([y_preds,y_pred], dim=0)
    y_true = torch.concat([y_true, y],dim=0)
# %%
y_preds = y_preds.cpu().numpy()
y_true = y_true.numpy()

mse_iTransformer = mean_squared_error(y_true.squeeze(), y_preds.squeeze())
mae_iTransformer = mean_absolute_error(y_true.squeeze(), y_preds.squeeze())
print(f'''Mean Squared Eror: {mse_transformer} \n
Mean Absolute Eror: {mae_transformer}''')
torch.save(iTransformer_model.state_dict(), "iTransformer_model.pt")
# %%