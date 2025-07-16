#!/usr/bin/env python3
"""
Demonstration script for OTB-100 tracking benchmark.
Uses Deer sequence (71 frames) for a quick complete demonstration.
Always performs a full run by clearing existing results first.
"""

import os
import json
import shutil
import time
from services.noise import apply_noise_and_occlusion_to_sequences, purge_noise_folders
from services.benchmark import run_trackers_and_evaluate
from services.analysis import plot_metrics, plot_eao_trends, plot_precision_vs_robustness, generate_metrics_table
from services.results import save_all_sequences_consolidated_csv

def main():
    """
    Demonstration of the complete tracking benchmark pipeline.
    Uses Deer sequence (shortest sequence with 71 frames) for quick demo.
    """
    print("🎯 OTB-100 Tracking Benchmark - DEMONSTRATION MODE")
    print("=" * 60)
    
    # Demo configuration
    DEMO_SEQUENCE = "Deer"
    DEMO_SEQUENCES = [DEMO_SEQUENCE]
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print(f"📁 Demo Sequence: {DEMO_SEQUENCE} (71 frames)")
    print(f"🎮 Trackers: {config['TRACKERS']}")
    print(f"🔊 Noise Types: {config['NOISE_TYPES']}")
    print(f"🔊 Occlusion Levels: {config['OCCLUSION_LEVELS']}")
    print(f"📊 Total Combinations: {len(config['TRACKERS']) * len(config['NOISE_TYPES']) * len(config['OCCLUSION_LEVELS'])}")
    print()
    
    # Step 1: Clear existing results for demo sequence
    print("🧹 STEP 1: Clearing existing results for fresh demo...")
    demo_results_dir = os.path.join(config['RESULTS_DIR'], DEMO_SEQUENCE)
    if os.path.exists(demo_results_dir):
        print(f"   Removing existing results: {demo_results_dir}")
        shutil.rmtree(demo_results_dir)
    else:
        print(f"   No existing results found for {DEMO_SEQUENCE}")
    
    # Step 2: Clear existing noise frames for demo sequence
    print("🔊 STEP 2: Clearing existing noise frames for fresh generation...")
    demo_dataset_dir = os.path.join(config['DATASET_PATH'], DEMO_SEQUENCE)
    noise_combinations = []
    for noise_type in config['NOISE_TYPES']:
        for occlusion_level in config['OCCLUSION_LEVELS']:
            noise_combinations.append(f"{noise_type}_{occlusion_level}")
    
    for combo in noise_combinations:
        noise_dir = os.path.join(demo_dataset_dir, f"img_{combo}")
        if os.path.exists(noise_dir):
            print(f"   Removing existing noise frames: img_{combo}")
            shutil.rmtree(noise_dir)
    
    print("✅ Cleanup completed - ready for fresh demo run!")
    print()
    
    # Initialize timing variables
    noise_time = 0
    tracking_time = 0
    analysis_time = 0
    
    # Step 3: Apply noise and occlusion
    print("🔊 STEP 3: Applying noise and occlusion to frames...")
    start_time = time.time()
    try:
        apply_noise_and_occlusion_to_sequences(
            dataset_path=config['DATASET_PATH'],
            sequences=DEMO_SEQUENCES,
            noise_types=config['NOISE_TYPES'],
            occlusion_levels=config['OCCLUSION_LEVELS']
        )
        noise_time = time.time() - start_time
        print(f"✅ Noise application completed in {noise_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error applying noise: {e}")
        return
    print()
    
    # Step 4: Run tracking evaluation
    print("🎯 STEP 4: Running tracking evaluation...")
    start_time = time.time()
    try:
        results = run_trackers_and_evaluate(
            dataset_path=config['DATASET_PATH'],
            results_dir=config['RESULTS_DIR'],
            sequences=DEMO_SEQUENCES,
            trackers=config['TRACKERS'],
            visualize=False,  # Disable visualization for demo
            save_videos=config['SAVE_VIDEOS'],
            failure_threshold=config['FAILURE_THRESHOLD'],
            frame_delay=config['FRAME_DELAY']
        )
        tracking_time = time.time() - start_time
        print(f"✅ Tracking evaluation completed in {tracking_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error during tracking: {e}")
        return
    print()
    
    # Step 5: Generate comprehensive analysis
    print("📊 STEP 5: Generating comprehensive analysis...")
    start_time = time.time()
    try:
        # Generate plots for the demo sequence
        plot_metrics(config['RESULTS_DIR'], DEMO_SEQUENCE)
        plot_eao_trends(config['RESULTS_DIR'], DEMO_SEQUENCE)
        plot_precision_vs_robustness(config['RESULTS_DIR'], DEMO_SEQUENCE)
        generate_metrics_table(config['RESULTS_DIR'], DEMO_SEQUENCE)
        
        # Generate consolidated CSV
        save_all_sequences_consolidated_csv(results, config['RESULTS_DIR'], sequences=DEMO_SEQUENCES)
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        analysis_time = 0
        print("⚠️  Continuing without analysis plots...")
    print()
    
    # Step 6: Demo results summary
    print("📋 STEP 6: Demo Results Summary")
    print("-" * 40)
    
    if DEMO_SEQUENCE in results and results[DEMO_SEQUENCE]:
        demo_results = results[DEMO_SEQUENCE]
        
        print(f"🎯 Sequence: {DEMO_SEQUENCE}")
        print(f"📦 Trackers evaluated: {len(demo_results)}")
        
        total_combinations = sum(len(tracker_results) for tracker_results in demo_results.values())
        print(f"🔄 Total combinations processed: {total_combinations}")
        
        # Show best performance for each tracker
        for tracker_name, tracker_results in demo_results.items():
            if tracker_results:
                best_combo = max(tracker_results.keys(), key=lambda k: tracker_results[k]['EAO'])
                best_eao = tracker_results[best_combo]['EAO']
                print(f"   🏆 {tracker_name} best EAO: {best_eao:.4f} (combo: {best_combo})")
        
        print(f"📁 Results saved to: {os.path.join(config['RESULTS_DIR'], DEMO_SEQUENCE)}")
        print(f"📊 Plots generated: metrics_bar.png, eao_trends.png, precision_vs_robustness.png")
        print(f"📋 CSV files: individual and consolidated metrics tables")
        
        if config['SAVE_VIDEOS']:
            print(f"🎬 Videos saved: tracking videos for each tracker/combination")
    else:
        print(f"⚠️  No results found for {DEMO_SEQUENCE} - check for errors above")
    
    # Total demo time
    total_time = noise_time + tracking_time + analysis_time
    print()
    print("🎉 DEMONSTRATION COMPLETED!")
    print(f"⏱️  Total demo time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if total_time > 0:
        print(f"📊 Processing rate: {71/total_time:.1f} frames/second")
    print()
    print("🔍 Check the results folder for:")
    print(f"   📁 {os.path.join(config['RESULTS_DIR'], DEMO_SEQUENCE)}/")
    print("   📊 CSV metrics tables")
    print("   📈 Performance visualization plots")
    if config['SAVE_VIDEOS']:
        print("   🎬 Tracking result videos")
    print()
    print("✨ This demo showcases the complete benchmark pipeline!")

if __name__ == "__main__":
    main()
