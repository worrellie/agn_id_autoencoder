import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
filename = 'wandb_export_2026-07-22T13_59_27.544+01_00.csv'
df = pd.read_csv(filename)

# Clean specific spreadsheet error values
df = df.replace(['#DIV/0!', '#VALUE!'], np.nan)

epoch_cols = [c for c in df.columns if 'epoch' in c.lower()]
best_epoch_col = epoch_cols[0] if epoch_cols else None

if best_epoch_col in df.columns:
    df[best_epoch_col] = pd.to_numeric(df[best_epoch_col], errors='coerce')

if 'latent_size' in df.columns:
    df['latent_size'] = pd.to_numeric(df['latent_size'], errors='coerce')

# Drop NaNs before plotting to avoid seaborn issues with nullable Int64 and NA
plot_df = df.dropna(subset=['activation', best_epoch_col, 'latent_size'], how='any').copy()
# Format as standard integers after dropping nans
plot_df['latent_size'] = plot_df['latent_size'].astype(int)

# Create a single figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.set_theme(style="whitegrid")

# Plot 1: Activation 
if 'activation' in plot_df.columns:
    sns.countplot(x='activation', data=plot_df, color='skyblue', ax=axes[0])
    axes[0].set_title('Distribution of Activation')
    axes[0].set_xlabel('Activation Function')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', )

# Plot 2: Best Epoch 
if best_epoch_col in plot_df.columns:
    sns.histplot(plot_df[best_epoch_col], kde=False, color='skyblue', bins=20, ax=axes[1])
    axes[1].set_title(f'Distribution of {best_epoch_col}')
    axes[1].set_xlabel('Best Epoch')
    axes[1].set_ylabel('Frequency')

# Plot 3: Latent Size 
if 'latent_size' in plot_df.columns:
    sns.countplot(x='latent_size', data=plot_df, color='skyblue', ax=axes[2])
    axes[2].set_title('Distribution of Latent Size')
    axes[2].set_xlabel('Latent Size')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x',)

plt.tight_layout()
plt.show()