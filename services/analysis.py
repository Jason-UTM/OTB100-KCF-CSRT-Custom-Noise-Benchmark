import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(results_dir: str, sequence: str):
    """
    Create bar plots for EAO, Robustness, and Precision by tracker and noise/occlusion.
    
    Args:
        results_dir: Directory containing results (e.g., 'results')
        sequence: Sequence name (e.g., 'Basketball')
    """
    # Load consolidated CSV
    seq_dir = os.path.join(results_dir, sequence)
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(f"Results directory {seq_dir} not found")
    
    csv_path = os.path.join(seq_dir, f"{sequence}_metrics_table.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    
    # Define combinations for ordering
    combinations = [f"{nt}_{ol}" for nt in ['gaussian', 'salt_pepper'] for ol in [0.2, 0.4, 0.6]]
    
    # Plot settings
    metrics = ['EAO', 'Robustness', 'Precision']
    titles = ['Expected Average Overlap (EAO)', 'Robustness (Failures/Frame)', 'Precision (Center Error)']
    ylabels = ['EAO', 'Robustness', 'Pixels']
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        # Pivot data for grouped bar plot
        pivot = df.pivot(index='Combo', columns='Tracker', values=metric)
        pivot = pivot.reindex(combinations)  # Ensure consistent order
        
        # Plot
        pivot.plot(kind='bar', ax=axes[i], width=0.4)
        axes[i].set_title(title)
        axes[i].set_xlabel('Noise/Occlusion')
        axes[i].set_ylabel(ylabel)
        axes[i].legend(title='Tracker')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, sequence, f'{sequence}_metrics_bar.png'))
    plt.show()

def plot_eao_trends(results_dir: str, sequence: str):
    """
    Create line plot for EAO vs. occlusion level by tracker and noise type.
    
    Args:
        results_dir: Directory containing results
        sequence: Sequence name
    """
    seq_dir = os.path.join(results_dir, sequence)
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(f"Results directory {seq_dir} not found")
    
    csv_path = os.path.join(seq_dir, f"{sequence}_metrics_table.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    
    # Parse combo column to extract noise and occlusion
    df['Noise'] = df['Combo'].str.rsplit('_', n=1).str[0]
    df['Occlusion'] = df['Combo'].str.rsplit('_', n=1).str[1].astype(float)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for tracker in df['Tracker'].unique():
        for noise in df['Noise'].unique():
            subset = df[(df['Tracker'] == tracker) & (df['Noise'] == noise)]
            subset = subset.sort_values('Occlusion')
            plt.plot(subset['Occlusion'], subset['EAO'], marker='o', label=f'{tracker} ({noise})')
    
    plt.title(f'EAO vs. Occlusion Level ({sequence})')
    plt.xlabel('Occlusion Level')
    plt.ylabel('EAO')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, sequence, f'{sequence}_eao_trends.png'))
    plt.show()

def generate_metrics_table(results_dir: str, sequence: str):
    """
    Generate a table of all metrics and save as Markdown (CSV already exists from benchmark).
    
    Args:
        results_dir: Directory containing results
        sequence: Sequence name
    """
    seq_dir = os.path.join(results_dir, sequence)
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(f"Results directory {seq_dir} not found")
    
    csv_path = os.path.join(seq_dir, f"{sequence}_metrics_table.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    
    # Parse combo column to extract noise and occlusion for sorting
    df['Noise'] = df['Combo'].str.rsplit('_', n=1).str[0]
    df['Occlusion'] = df['Combo'].str.rsplit('_', n=1).str[1].astype(float)
    
    # Sort by tracker, noise, and occlusion
    df = df.sort_values(['Tracker', 'Noise', 'Occlusion'])
    
    # Drop temporary columns for display
    display_df = df[['Tracker', 'Combo', 'EAO', 'Robustness', 'Precision', 
                     'TrackingTime', 'FPS', 'NumFrames', 'NumFailures']]
    
    # Save as Markdown
    markdown = display_df.to_markdown(index=False)
    markdown_path = os.path.join(seq_dir, f'{sequence}_metrics_table.md')
    with open(markdown_path, 'w') as f:
        f.write(f"# Metrics Table for {sequence}\n\n{markdown}")
    print(f"Saved Markdown table to {markdown_path}")

def plot_precision_vs_robustness(results_dir: str, sequence: str):
    """
    Create scatter plot of Precision vs. Robustness by tracker and occlusion level.
    
    Args:
        results_dir: Directory containing results
        sequence: Sequence name
    """
    seq_dir = os.path.join(results_dir, sequence)
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(f"Results directory {seq_dir} not found")
    
    csv_path = os.path.join(seq_dir, f"{sequence}_metrics_table.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated metrics file {csv_path} not found")
    
    df = pd.read_csv(csv_path)
    
    # Parse combo column to extract occlusion
    df['Occlusion'] = df['Combo'].str.rsplit('_', n=1).str[1].astype(float)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for tracker in df['Tracker'].unique():
        subset = df[df['Tracker'] == tracker]
        plt.scatter(subset['Robustness'], subset['Precision'], 
                   s=subset['Occlusion']*200, label=tracker, alpha=0.6)
    
    plt.title(f'Precision vs. Robustness ({sequence})')
    plt.xlabel('Robustness (Failures/Frame)')
    plt.ylabel('Precision (Center Error, Pixels)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, sequence, f'{sequence}_precision_vs_robustness.png'))
    plt.show()