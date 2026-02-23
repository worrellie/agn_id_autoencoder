import torch
from torch import nn

def loss_calc(x_hat, x, mu, logvar, beta=1, red = 'mean'):

    n_pixels = len(x[0])
    batch_size = len(x)
    n_elements = n_pixels * batch_size

    recon_loss = nn.MSELoss(reduction = red)

    mse = recon_loss(x_hat, x)
    if red == 'mean':
        kl = -(0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) # sum over batch and latent dim
        kl = kl/n_elements # mean kl per pixel (to match 'mean' of MSE)
    elif red == 'sum':
        kl = -(0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    loss = mse + (beta * kl)

    return mse, kl, loss

def train_ae(epochs, train_loader, valid_loader, model, optimizer, verbose = True, *args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('training model...')

    model.to(device)

    # recon_loss = nn.MSELoss()

    train_losses = []
    valid_losses = []
    mses = []
    kls = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        valid_loss = 0

        for x, _ in train_loader:

            x = x.to(device)

            x_hat, mu, logvar = model(x) # batch prediction

            mse, kl, loss = loss_calc(x_hat, x, mu, logvar,)

            optimizer.zero_grad()

            loss.backward()

            train_loss += loss.item() * x.size(0) # cumulative loss of batch * size of batch
            # mse and kl

            optimizer.step()

        epoch_avg_loss = train_loss / len(train_loader.dataset) # average loss per sample for that epoch
        train_losses.append(epoch_avg_loss) # losses for each epoch

        if verbose:
            print(f'training: epoch {epoch+1}/{epochs}, loss: {epoch_avg_loss:.10f}')

        
        if valid_loader is not None:

            model.eval()

            with torch.no_grad():

                for x, _ in valid_loader:

                    x = x.to(device)

                    x_hat, mu, logvar = model(x)

                    loss = loss_calc(x_hat, x, mu, logvar,)

                    
                    valid_loss += loss.item()
                
                epoch_avg_valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(epoch_avg_valid_loss)

            if verbose:
                print(f'valid: epoch {epoch+1}/{epochs}, loss: {epoch_avg_valid_loss:.10f}')
        
    print('training finished')

    return model, train_losses, valid_losses


def predict(model, x):

    model.eval()

    losses = []
    reconstructed = []
    with torch.no_grad():

        # recon_loss = nn.MSELoss()

        total_loss = 0
        for x, _ in loader:

            x = x.to(device)

            x_hat, mu, logvar = model(x)

            loss = loss_calc(x_hat, x, mu, logvar,)

            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader.dataset)
        losses.append(avg_loss)

    return losses, 