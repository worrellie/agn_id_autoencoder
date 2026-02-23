import torch
from torch import nn, optim

class VAEAutoencoder(nn.Module):

    def __init__(self, config, input_size, latent_size):
        super(VAEAutoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.input_to_encoder = nn.LazyLinear(config[0]['in'])

        # add encoder layers
        for c in config:
            self.encoder_layers.append(
                nn.Linear(c['in'], c['out'], )
            )

    # ###### this is cool- remember for future
    # def _get_flattened_size(self, input_size):
        
    #     with torch.no_grad(): # do not update weights
            
    #         dummy_x = torch.zeros(1, 1, input_size)
    #         for l in self.encoder_layers:
    #             dummy_x = l(dummy_x) # updates the dummy shape based on the encoder layers
            
    #         return dummy_x.numel(), dummy_x.shape[1], dummy_x.shape[2]
    # ######
        
        
        # add decoder layers
        for c in reversed(config):
            self.decoder_layers.append(
                nn.Linear(c['out'], c['in'], )
            )
        
        # add latent layers
        self.encoder_to_latent_mean = nn.LazyLinear(latent_size)
        self.encoder_to_latent_logvar = nn.LazyLinear(latent_size)
        
        self.decoder_from_latent = nn.LazyLinear(config[-1]['out'])

        self.decoder_to_output = nn.LazyLinear(input_size)


    def forward(self, x):
        
        # print(x.shape)

        x = torch.relu(self.input_to_encoder(x))
        # print(x.shape)

        for l in self.encoder_layers:
            x = torch.relu(l(x))
            # print(x.shape)

        mu = torch.relu(self.encoder_to_latent_mean(x))
        logvar = torch.relu(self.encoder_to_latent_logvar(x))
        # print(mu.shape)
        # print(logvar.shape)

        epsilon = torch.randn_like(logvar)
        z = mu + logvar*epsilon # latent of VAE
        # print(z.shape)

        z = torch.relu(self.decoder_from_latent(z))

        for l in self.decoder_layers:
            z = torch.relu(l(z))
            # print(z.shape)

        # x_hat = torch.relu(self.decoder_to_output(z))
        # x_hat = torch.tanh(self.decoder_to_output(z))
        x_hat = self.decoder_to_output(z)
        # print(x_hat.shape)

        return x_hat, mu, logvar
