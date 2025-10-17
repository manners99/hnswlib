#!/usr/bin/env python3
"""
HNSW Degradation Test Comparison Visualizer
Compares results from two different degradation test CSV files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def compare_results(csv_file1, csv_file2, label1="Method 1", label2="Method 2", output_dir="comparison"):
    """Compare degradation test results from two CSV files"""
    
    # Read the CSV data
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (15, 10)
    
    # 1. Recall Comparison
    plt.figure(figsize=fig_size)
    plt.subplot(2, 3, 1)
    plt.plot(df1['iteration'], df1['recall'], 'b-', linewidth=2, label=label1, marker='o', markersize=4)
    plt.plot(df2['iteration'], df2['recall'], 'r-', linewidth=2, label=label2, marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.title('Recall Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Disconnected Nodes Comparison
    plt.subplot(2, 3, 2)
    plt.plot(df1['iteration'], df1['disconnected_nodes'], 'b-', linewidth=2, label=label1, marker='o', markersize=4)
    plt.plot(df2['iteration'], df2['disconnected_nodes'], 'r-', linewidth=2, label=label2, marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Disconnected Nodes')
    plt.title('Disconnected Nodes Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Average Inbound Connections
    plt.subplot(2, 3, 3)
    plt.plot(df1['iteration'], df1['avg_inbound'], 'b-', linewidth=2, label=label1, marker='o', markersize=4)
    plt.plot(df2['iteration'], df2['avg_inbound'], 'r-', linewidth=2, label=label2, marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Avg Inbound Connections')
    plt.title('Average Inbound Connections')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Connectivity Density
    plt.subplot(2, 3, 4)
    plt.plot(df1['iteration'], df1['connectivity_density'], 'b-', linewidth=2, label=label1, marker='o', markersize=4)
    plt.plot(df2['iteration'], df2['connectivity_density'], 'r-', linewidth=2, label=label2, marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Connectivity Density')
    plt.title('Connectivity Density Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 5. Search Time Comparison
    plt.subplot(2, 3, 5)
    plt.plot(df1['iteration'], df1['search_time_ms'], 'b-', linewidth=2, label=label1, marker='o', markersize=4)
    plt.plot(df2['iteration'], df2['search_time_ms'], 'r-', linewidth=2, label=label2, marker='s', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Time Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 6. Recall vs Disconnected Nodes (Scatter)
    plt.subplot(2, 3, 6)
    plt.scatter(df1['disconnected_nodes'], df1['recall'], c='blue', alpha=0.7, label=label1, s=50)
    plt.scatter(df2['disconnected_nodes'], df2['recall'], c='red', alpha=0.7, label=label2, s=50)
    plt.xlabel('Disconnected Nodes')
    plt.ylabel('Recall')
    plt.title('Recall vs Disconnected Nodes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, 'comparison_overview.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed comparison statistics
    stats_file = os.path.join(output_dir, 'comparison_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("HNSW Degradation Test Comparison Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        # Final iteration comparison
        final_iter = min(df1['iteration'].max(), df2['iteration'].max())
        final1 = df1[df1['iteration'] == final_iter].iloc[0]
        final2 = df2[df2['iteration'] == final_iter].iloc[0]
        
        f.write(f"Final Iteration ({final_iter}) Comparison:\n")
        f.write(f"{label1:20s} {label2:20s} Difference\n")
        f.write("-" * 50 + "\n")
        
        metrics = [
            ('Recall', 'recall', '.4f'),
            ('Disconnected Nodes', 'disconnected_nodes', '.0f'),
            ('Avg Inbound Conn', 'avg_inbound', '.2f'),
            ('Connectivity Density', 'connectivity_density', '.4f'),
            ('Search Time (ms)', 'search_time_ms', '.1f'),
            ('Total Connections', 'total_connections', '.0f')
        ]
        
        improvements = []
        for metric_name, metric_col, fmt in metrics:
            val1 = float(final1[metric_col])
            val2 = float(final2[metric_col])
            diff = val2 - val1
            if metric_col == 'disconnected_nodes' or metric_col == 'search_time_ms':
                # Lower is better for these metrics
                improvement = -diff
            else:
                # Higher is better for these metrics
                improvement = diff
            
            improvements.append((metric_name, improvement, val1, val2, diff))
            f.write(f"{metric_name:15s}: {val1:{fmt}} vs {val2:{fmt}} (diff: {diff:+{fmt}})\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Summary of Improvements:\n")
        f.write("-" * 25 + "\n")
        
        for metric_name, improvement, val1, val2, diff in improvements:
            if abs(improvement) > 0.001:  # Only show meaningful improvements
                direction = "better" if improvement > 0 else "worse"
                f.write(f"{label2} is {direction} in {metric_name}: {improvement:+.3f}\n")
        
        # Statistical analysis
        f.write("\n" + "=" * 50 + "\n")
        f.write("Statistical Analysis:\n")
        f.write("-" * 20 + "\n")
        
        recall_improvement = (final2['recall'] - final1['recall']) / final1['recall'] * 100
        disconnected_reduction = (final1['disconnected_nodes'] - final2['disconnected_nodes']) / final1['disconnected_nodes'] * 100
        
        f.write(f"Recall improvement: {recall_improvement:+.2f}%\n")
        f.write(f"Disconnected nodes reduction: {disconnected_reduction:+.2f}%\n")
        
        # Trend analysis
        if len(df1) > 5 and len(df2) > 5:
            # Calculate trends from iteration 1 onwards
            trend_data1 = df1[df1['iteration'] >= 1]
            trend_data2 = df2[df2['iteration'] >= 1]
            
            if len(trend_data1) > 1 and len(trend_data2) > 1:
                recall_trend1 = np.polyfit(trend_data1['iteration'], trend_data1['recall'], 1)[0]
                recall_trend2 = np.polyfit(trend_data2['iteration'], trend_data2['recall'], 1)[0]
                
                f.write(f"\nRecall trend analysis:\n")
                f.write(f"{label1} recall trend: {recall_trend1:+.5f} per iteration\n")
                f.write(f"{label2} recall trend: {recall_trend2:+.5f} per iteration\n")
                
                if abs(recall_trend2) < abs(recall_trend1):
                    f.write(f"{label2} shows more stable recall over time\n")
                else:
                    f.write(f"{label1} shows more stable recall over time\n")
    
    # Create individual focused comparison plots
    
    # Focused Recall Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df1['iteration'], df1['recall'], 'b-', linewidth=3, label=label1, marker='o', markersize=6)
    plt.plot(df2['iteration'], df2['recall'], 'r-', linewidth=3, label=label2, marker='s', markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Search Recall Comparison Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    recall_file = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(recall_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Focused Disconnected Nodes Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df1['iteration'], df1['disconnected_nodes'], 'b-', linewidth=3, label=label1, marker='o', markersize=6)
    plt.plot(df2['iteration'], df2['disconnected_nodes'], 'r-', linewidth=3, label=label2, marker='s', markersize=6)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Disconnected Nodes', fontsize=12)
    plt.title('Disconnected Nodes Comparison Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    disconnected_file = os.path.join(output_dir, 'disconnected_comparison.png')
    plt.savefig(disconnected_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison analysis completed!")
    print(f"Output directory: {output_dir}/")
    print("Generated files:")
    print("  - comparison_overview.png")
    print("  - recall_comparison.png") 
    print("  - disconnected_comparison.png")
    print("  - comparison_statistics.txt")

def main():
    parser = argparse.ArgumentParser(description='Compare two HNSW degradation test CSV files')
    parser.add_argument('csv_file1', help='Path to the first CSV file')
    parser.add_argument('csv_file2', help='Path to the second CSV file')
    parser.add_argument('--label1', default='Method 1', help='Label for first method (default: Method 1)')
    parser.add_argument('--label2', default='Method 2', help='Label for second method (default: Method 2)')
    parser.add_argument('-o', '--output', default='comparison', help='Output directory for comparison graphs (default: comparison)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file1):
        print(f"Error: File {args.csv_file1} does not exist")
        return 1
        
    if not os.path.exists(args.csv_file2):
        print(f"Error: File {args.csv_file2} does not exist")
        return 1
    
    compare_results(args.csv_file1, args.csv_file2, args.label1, args.label2, args.output)
    return 0

if __name__ == "__main__":
    sys.exit(main())
