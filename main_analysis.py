from services.config import load_config
from services.analysis import plot_metrics, plot_eao_trends, plot_precision_vs_robustness, overall_performance

if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = r".\config.json"
    config = load_config(CONFIG_PATH)

    RESULTS_DIR = config['RESULTS_DIR']
    SEQUENCES = config['SEQUENCES']
    
    # Analyze results
    for sequence in SEQUENCES:
        print(f"Analyzing results for sequence: {sequence}")
        plot_metrics(RESULTS_DIR, sequence)
        plot_eao_trends(RESULTS_DIR, sequence)
        plot_precision_vs_robustness(RESULTS_DIR, sequence)
    
    # Overall performance across all sequences
    overall_performance()
    

'''
EAO: Higher values indicate better tracking accuracy. Compare KCF vs. CSRT to see which maintains higher overlap under increasing occlusion (e.g., CSRT may outperform KCF due to its robustness to partial occlusions).
Robustness: Lower values mean fewer failures. Check if salt_pepper noise causes more failures than gaussian due to its sparse, high-contrast nature.
Precision: Lower pixel errors indicate better localization. Assess if precision degrades significantly at higher occlusion levels (e.g., 0.6).
'''