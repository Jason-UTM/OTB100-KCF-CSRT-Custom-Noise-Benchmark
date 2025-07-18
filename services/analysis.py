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
    # Check if plot already exists
    plot_path = os.path.join(results_dir, sequence, f'{sequence}_metrics_bar.png')
    if os.path.exists(plot_path):
        print(f"Plot already exists, skipping: {plot_path}")
        return
    
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
    # Check if plot already exists
    plot_path = os.path.join(results_dir, sequence, f'{sequence}_eao_trends.png')
    if os.path.exists(plot_path):
        print(f"Plot already exists, skipping: {plot_path}")
        return
    
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
    # Check if plot already exists
    plot_path = os.path.join(results_dir, sequence, f'{sequence}_precision_vs_robustness.png')
    if os.path.exists(plot_path):
        print(f"Plot already exists, skipping: {plot_path}")
        return
    
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

def overall_performance():
    """
    Caculate overall performance metrics grouped by tracker from the consolidated CSV with all sequences.
    """
    # Load consolidated CSV
    csv_path = os.path.join('results', 'all_sequences_metrics_table.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated metrics file {csv_path} not found")
    df = pd.read_csv(csv_path)
    # Group by tracker and calculate mean metrics
    overall = df.groupby('Tracker').agg({
        'EAO': 'mean',
        'Robustness': 'mean',
        'Precision': 'mean',
        'TrackingTime': 'mean', 
        'FPS': 'mean',
        'NumFrames': 'sum',
        'NumFailures': 'sum'
    }).reset_index()
    print("Overall performance metrics:")
    print(overall)

    overall_by_tracker_and_combo = df.groupby(['Tracker', 'Combo']).agg({
        'EAO': 'mean',
        'Robustness': 'mean',
        'Precision': 'mean',
        'TrackingTime': 'mean',
        'FPS': 'mean',
        'NumFrames': 'sum',
        'NumFailures': 'sum'
    }).reset_index()
    print("Overall performance metrics by tracker and combo:")
    print(overall_by_tracker_and_combo)

    # Save overall performance to CSV
    overall_csv_path = os.path.join('results', 'overall_performance.csv')
    overall.to_csv(overall_csv_path, index=False)
    print(f"Saved overall performance metrics to {overall_csv_path}")

    # Save overall performance by tracker and combo to CSV
    overall_by_combo_csv_path = os.path.join('results', 'overall_performance_by_combo.csv')
    overall_by_tracker_and_combo.to_csv(overall_by_combo_csv_path, index=False)
    print(f"Saved overall performance by combo metrics to {overall_by_combo_csv_path}")

    # Plot settings - all metrics from aggregation
    metrics = ['EAO', 'Robustness', 'Precision', 'TrackingTime', 'FPS', 'NumFrames', 'NumFailures']
    titles = ['Expected Average Overlap (EAO)', 'Robustness (Failures/Frame)', 'Precision (Center Error)', 
              'Average Tracking Time', 'Average FPS', 'Total Frames', 'Total Failures']
    ylabels = ['EAO', 'Failures/Frame', 'Pixels', 'Seconds', 'FPS', 'Frames', 'Failures']
    
    # ===== PLOT 1: Overall Performance by Tracker =====
    # Create subplots in a single figure - 3 rows, 3 columns (with one empty subplot)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten to easily iterate
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        # Plot bar chart for each metric
        axes[i].bar(overall['Tracker'], overall[metric])
        axes[i].set_title(title)
        axes[i].set_ylabel(ylabel)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide the last two empty subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'overall_performance.png'))
    plt.show()
    print(f"Saved overall performance plot to {os.path.join('results', 'overall_performance.png')}")

    # ===== PLOT 2: Overall Performance by Tracker and Combo =====
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    axes = axes.flatten()  # Flatten to easily iterate
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        # Pivot data for grouped bar plot
        pivot = overall_by_tracker_and_combo.pivot(index='Combo', columns='Tracker', values=metric)
        
        # Plot
        pivot.plot(kind='bar', ax=axes[i], width=0.6)
        axes[i].set_title(f'{title} by Tracker and Combo')
        axes[i].set_xlabel('Noise/Occlusion')
        axes[i].set_ylabel(ylabel)
        axes[i].legend(title='Tracker')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide the last two empty subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'overall_performance_by_combo.png'))
    plt.show()
    print(f"Saved overall performance by combo plot to {os.path.join('results', 'overall_performance_by_combo.png')}")

    # ===== PLOT 3: EAO Trends for Overall Performance =====
    # Parse combo column to extract noise and occlusion for overall_by_tracker_and_combo
    overall_by_tracker_and_combo['Noise'] = overall_by_tracker_and_combo['Combo'].str.rsplit('_', n=1).str[0]
    overall_by_tracker_and_combo['Occlusion'] = overall_by_tracker_and_combo['Combo'].str.rsplit('_', n=1).str[1].astype(float)
    
    plt.figure(figsize=(12, 8))
    for tracker in overall_by_tracker_and_combo['Tracker'].unique():
        for noise in overall_by_tracker_and_combo['Noise'].unique():
            subset = overall_by_tracker_and_combo[(overall_by_tracker_and_combo['Tracker'] == tracker) & 
                                                (overall_by_tracker_and_combo['Noise'] == noise)]
            subset = subset.sort_values('Occlusion')
            plt.plot(subset['Occlusion'], subset['EAO'], marker='o', linewidth=2, markersize=8, 
                    label=f'{tracker} ({noise})')
    
    plt.title('Overall EAO vs. Occlusion Level (All Sequences)')
    plt.xlabel('Occlusion Level')
    plt.ylabel('EAO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('results', 'overall_eao_trends.png'))
    plt.show()
    print(f"Saved overall EAO trends plot to {os.path.join('results', 'overall_eao_trends.png')}")

    # ===== PLOT 4: Precision vs Robustness for Overall Performance =====
    # Plot 4a: Overall by tracker only
    plt.figure(figsize=(10, 6))
    for tracker in overall['Tracker'].unique():
        subset = overall[overall['Tracker'] == tracker]
        plt.scatter(subset['Robustness'], subset['Precision'], s=200, label=tracker, alpha=0.7)
    
    plt.title('Overall Precision vs. Robustness (All Sequences)')
    plt.xlabel('Robustness (Failures/Frame)')
    plt.ylabel('Precision (Center Error, Pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('results', 'overall_precision_vs_robustness.png'))
    plt.show()
    print(f"Saved overall precision vs robustness plot to {os.path.join('results', 'overall_precision_vs_robustness.png')}")

    # Plot 4b: Overall by tracker and combo
    plt.figure(figsize=(12, 8))
    for tracker in overall_by_tracker_and_combo['Tracker'].unique():
        subset = overall_by_tracker_and_combo[overall_by_tracker_and_combo['Tracker'] == tracker]
        plt.scatter(subset['Robustness'], subset['Precision'], 
                   s=subset['Occlusion']*300, label=tracker, alpha=0.6)
    
    plt.title('Overall Precision vs. Robustness by Combo (All Sequences)')
    plt.xlabel('Robustness (Failures/Frame)')
    plt.ylabel('Precision (Center Error, Pixels)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('results', 'overall_precision_vs_robustness_by_combo.png'))
    plt.show()
    print(f"Saved overall precision vs robustness by combo plot to {os.path.join('results', 'overall_precision_vs_robustness_by_combo.png')}")

    return overall
