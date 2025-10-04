#!/usr/bin/env python3
"""
HNSW Degradation Test Results Visualizer
Generates graphs from the CSV output of the degradation test
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def create_graphs(csv_file, output_dir="graphs"):
    """Create various graphs from the degradation test CSV data"""
    
    # Read the CSV data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Recall over time
    plt.figure(figsize=fig_size)
    plt.plot(df['iteration'], df['recall'], 'b-', linewidth=2, label='Recall')
    
    # Add trendline from iteration 1 onwards
    if len(df) > 1:
        trend_data = df[df['iteration'] >= 1]
        if len(trend_data) > 1:
            z = np.polyfit(trend_data['iteration'], trend_data['recall'], 1)
            p = np.poly1d(z)
            plt.plot(trend_data['iteration'], p(trend_data['iteration']), 'r--', 
                    linewidth=2, alpha=0.8, label=f'Trend (iter 1+): slope={z[0]:.4f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.title('Search Recall Degradation Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graph connectivity metrics
    plt.figure(figsize=fig_size)
    plt.subplot(2, 2, 1)
    plt.plot(df['iteration'], df['total_connections'], 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total Connections')
    plt.title('Total Graph Connections')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df['iteration'], df['disconnected_nodes'], 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Disconnected Nodes')
    plt.title('Disconnected Nodes')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(df['iteration'], df['avg_inbound'], 'purple', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Average Inbound Connections')
    plt.title('Average Inbound Connections')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(df['iteration'], df['connectivity_density'], 'orange', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Connectivity Density')
    plt.title('Graph Connectivity Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/connectivity_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance metrics
    plt.figure(figsize=fig_size)
    plt.subplot(2, 1, 1)
    plt.plot(df['iteration'], df['search_time_ms'], 'cyan', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Performance Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['iteration'], df['iteration_time_seconds'], 'magenta', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Iteration Time (seconds)')
    plt.title('Per-Iteration Processing Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Element counts
    plt.figure(figsize=fig_size)
    plt.plot(df['iteration'], df['active_elements'], 'g-', linewidth=2, label='Active Elements')
    plt.plot(df['iteration'], df['deleted_elements'], 'r-', linewidth=2, label='Deleted Elements')
    plt.xlabel('Iteration')
    plt.ylabel('Element Count')
    plt.title('Active vs Deleted Elements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/element_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Inbound connection distribution
    plt.figure(figsize=fig_size)
    plt.fill_between(df['iteration'], df['min_inbound'], df['max_inbound'], 
                     alpha=0.3, color='lightblue', label='Min-Max Range')
    plt.plot(df['iteration'], df['avg_inbound'], 'b-', linewidth=2, label='Average')
    plt.plot(df['iteration'], df['min_inbound'], 'r--', linewidth=1, label='Minimum')
    plt.plot(df['iteration'], df['max_inbound'], 'g--', linewidth=1, label='Maximum')
    plt.xlabel('Iteration')
    plt.ylabel('Inbound Connections')
    plt.title('Inbound Connection Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inbound_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Combined degradation overview
    plt.figure(figsize=(16, 10))
    
    # Create subplots with shared x-axis
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(df['iteration'], df['recall'], 'b-', linewidth=2, label='Recall')
    
    # Add trendline from iteration 1 onwards
    if len(df) > 1:
        trend_data = df[df['iteration'] >= 1]
        if len(trend_data) > 1:
            z = np.polyfit(trend_data['iteration'], trend_data['recall'], 1)
            p = np.poly1d(z)
            plt.plot(trend_data['iteration'], p(trend_data['iteration']), 'r--', 
                    linewidth=1.5, alpha=0.8, label=f'Trend: {z[0]:.4f}/iter')
    
    plt.ylabel('Recall')
    plt.title('Recall Degradation')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    ax2 = plt.subplot(3, 2, 2)
    plt.plot(df['iteration'], df['disconnected_nodes'], 'r-', linewidth=2)
    plt.ylabel('Disconnected Nodes')
    plt.title('Graph Fragmentation')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 2, 3)
    plt.plot(df['iteration'], df['total_connections'], 'g-', linewidth=2)
    plt.ylabel('Total Connections')
    plt.title('Graph Connectivity')
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 2, 4)
    plt.plot(df['iteration'], df['search_time_ms'], 'cyan', linewidth=2)
    plt.ylabel('Search Time (ms)')
    plt.title('Search Performance')
    plt.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 2, 5)
    plt.plot(df['iteration'], df['connectivity_density'], 'orange', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Connectivity Density')
    plt.title('Graph Density')
    plt.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 2, 6)
    plt.plot(df['iteration'], df['avg_inbound'], 'purple', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Avg Inbound Connections')
    plt.title('Connection Balance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degradation_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary statistics
    summary_file = f"{output_dir}/summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("HNSW Degradation Test Summary\n")
        f.write("============================\n\n")
        f.write(f"Total iterations: {len(df)}\n")
        f.write(f"Initial recall: {df['recall'].iloc[0]:.4f}\n")
        f.write(f"Final recall: {df['recall'].iloc[-1]:.4f}\n")
        f.write(f"Recall degradation: {df['recall'].iloc[0] - df['recall'].iloc[-1]:.4f}\n")
        f.write(f"Max disconnected nodes: {df['disconnected_nodes'].max()}\n")
        f.write(f"Initial connections: {df['total_connections'].iloc[0]}\n")
        f.write(f"Final connections: {df['total_connections'].iloc[-1]}\n")
        f.write(f"Connection change: {df['total_connections'].iloc[-1] - df['total_connections'].iloc[0]}\n")
        f.write(f"Average search time: {df['search_time_ms'].mean():.2f} ms\n")
        f.write(f"Total test time: {df['cumulative_time_seconds'].iloc[-1]:.2f} seconds\n")
        
        # Identify critical points
        max_disconnected_iter = df.loc[df['disconnected_nodes'].idxmax(), 'iteration']
        min_recall_iter = df.loc[df['recall'].idxmin(), 'iteration']
        
        f.write(f"\nCritical Points:\n")
        f.write(f"Maximum disconnected nodes at iteration: {max_disconnected_iter}\n")
        f.write(f"Minimum recall at iteration: {min_recall_iter}\n")
    
    print(f"Graphs generated successfully!")
    print(f"Output directory: {output_dir}/")
    print(f"Generated files:")
    print(f"  - recall_over_time.png")
    print(f"  - connectivity_metrics.png")
    print(f"  - performance_metrics.png")
    print(f"  - element_counts.png")
    print(f"  - inbound_distribution.png")
    print(f"  - degradation_overview.png")
    print(f"  - summary_statistics.txt")

def create_comparison_graphs(csv_file1, csv_file2, output_dir="comparison_graphs", 
                           label1="Without LSH", label2="With LSH"):
    """Create comparison graphs between two degradation test CSV files"""
    
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
    fig_size = (14, 10)
    
    # 1. Recall comparison
    plt.figure(figsize=fig_size)
    plt.plot(df1['iteration'], df1['recall'], 'b-', linewidth=2, label=label1, alpha=0.8)
    plt.plot(df2['iteration'], df2['recall'], 'r-', linewidth=2, label=label2, alpha=0.8)
    
    # Add trendlines
    for df, color, label in [(df1, 'blue', label1), (df2, 'red', label2)]:
        if len(df) > 1:
            trend_data = df[df['iteration'] >= 1]
            if len(trend_data) > 1:
                z = np.polyfit(trend_data['iteration'], trend_data['recall'], 1)
                p = np.poly1d(z)
                plt.plot(trend_data['iteration'], p(trend_data['iteration']), 
                        color=color, linestyle='--', linewidth=1.5, alpha=0.6,
                        label=f'{label} trend: {z[0]:.4f}/iter')
    
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.title('Search Recall Comparison: With vs Without LSH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Disconnected nodes comparison
    plt.figure(figsize=fig_size)
    plt.plot(df1['iteration'], df1['disconnected_nodes'], 'b-', linewidth=2, 
             label=f'{label1} (max: {df1["disconnected_nodes"].max()})', alpha=0.8)
    plt.plot(df2['iteration'], df2['disconnected_nodes'], 'r-', linewidth=2, 
             label=f'{label2} (max: {df2["disconnected_nodes"].max()})', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Disconnected Nodes')
    plt.title('Graph Fragmentation Comparison: With vs Without LSH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/disconnected_nodes_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Total connections comparison
    plt.figure(figsize=fig_size)
    plt.plot(df1['iteration'], df1['total_connections'], 'b-', linewidth=2, label=label1, alpha=0.8)
    plt.plot(df2['iteration'], df2['total_connections'], 'r-', linewidth=2, label=label2, alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Total Connections')
    plt.title('Graph Connectivity Comparison: With vs Without LSH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_connections_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Connectivity density comparison
    plt.figure(figsize=fig_size)
    plt.plot(df1['iteration'], df1['connectivity_density'], 'b-', linewidth=2, label=label1, alpha=0.8)
    plt.plot(df2['iteration'], df2['connectivity_density'], 'r-', linewidth=2, label=label2, alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Connectivity Density')
    plt.title('Graph Density Comparison: With vs Without LSH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/connectivity_density_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Search time comparison
    plt.figure(figsize=fig_size)
    plt.plot(df1['iteration'], df1['search_time_ms'], 'b-', linewidth=2, 
             label=f'{label1} (avg: {df1["search_time_ms"].mean():.1f}ms)', alpha=0.8)
    plt.plot(df2['iteration'], df2['search_time_ms'], 'r-', linewidth=2, 
             label=f'{label2} (avg: {df2["search_time_ms"].mean():.1f}ms)', alpha=0.8)
    plt.xlabel('Iteration')
    plt.ylabel('Search Time (ms)')
    plt.title('Search Performance Comparison: With vs Without LSH')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/search_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Comprehensive comparison dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Recall
    axes[0, 0].plot(df1['iteration'], df1['recall'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[0, 0].plot(df2['iteration'], df2['recall'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[0, 0].set_ylabel('Recall')
    axes[0, 0].set_title('Search Recall')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # Disconnected nodes
    axes[0, 1].plot(df1['iteration'], df1['disconnected_nodes'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[0, 1].plot(df2['iteration'], df2['disconnected_nodes'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[0, 1].set_ylabel('Disconnected Nodes')
    axes[0, 1].set_title('Graph Fragmentation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # Total connections
    axes[0, 2].plot(df1['iteration'], df1['total_connections'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[0, 2].plot(df2['iteration'], df2['total_connections'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[0, 2].set_ylabel('Total Connections')
    axes[0, 2].set_title('Graph Connectivity')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=10)
    
    # Connectivity density
    axes[1, 0].plot(df1['iteration'], df1['connectivity_density'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[1, 0].plot(df2['iteration'], df2['connectivity_density'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Connectivity Density')
    axes[1, 0].set_title('Graph Density')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    
    # Search time
    axes[1, 1].plot(df1['iteration'], df1['search_time_ms'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[1, 1].plot(df2['iteration'], df2['search_time_ms'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Search Time (ms)')
    axes[1, 1].set_title('Search Performance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    # Average inbound connections
    axes[1, 2].plot(df1['iteration'], df1['avg_inbound'], 'b-', linewidth=2, label=label1, alpha=0.8)
    axes[1, 2].plot(df2['iteration'], df2['avg_inbound'], 'r-', linewidth=2, label=label2, alpha=0.8)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Avg Inbound Connections')
    axes[1, 2].set_title('Connection Balance')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=10)
    
    plt.suptitle('HNSW Performance: With vs Without LSH Repair', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comparison summary
    summary_file = f"{output_dir}/comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("HNSW Performance Comparison: With vs Without LSH\n")
        f.write("================================================\n\n")
        f.write(f"Dataset 1 ({label1}):\n")
        f.write(f"  Final recall: {df1['recall'].iloc[-1]:.4f}\n")
        f.write(f"  Max disconnected nodes: {df1['disconnected_nodes'].max()}\n")
        f.write(f"  Average search time: {df1['search_time_ms'].mean():.2f} ms\n")
        f.write(f"  Final connections: {df1['total_connections'].iloc[-1]}\n")
        f.write(f"  Final connectivity density: {df1['connectivity_density'].iloc[-1]:.4f}\n\n")
        
        f.write(f"Dataset 2 ({label2}):\n")
        f.write(f"  Final recall: {df2['recall'].iloc[-1]:.4f}\n")
        f.write(f"  Max disconnected nodes: {df2['disconnected_nodes'].max()}\n")
        f.write(f"  Average search time: {df2['search_time_ms'].mean():.2f} ms\n")
        f.write(f"  Final connections: {df2['total_connections'].iloc[-1]}\n")
        f.write(f"  Final connectivity density: {df2['connectivity_density'].iloc[-1]:.4f}\n\n")
        
        # Calculate improvements
        recall_diff = df2['recall'].iloc[-1] - df1['recall'].iloc[-1]
        disconnected_diff = df1['disconnected_nodes'].max() - df2['disconnected_nodes'].max()
        time_diff = df1['search_time_ms'].mean() - df2['search_time_ms'].mean()
        
        f.write("Performance Differences:\n")
        f.write(f"  Recall improvement: {recall_diff:+.4f}\n")
        f.write(f"  Disconnected nodes reduction: {disconnected_diff:+d}\n")
        f.write(f"  Search time change: {time_diff:+.2f} ms\n")
    
    print(f"Comparison graphs generated successfully!")
    print(f"Output directory: {output_dir}/")
    print(f"Generated files:")
    print(f"  - recall_comparison.png")
    print(f"  - disconnected_nodes_comparison.png")
    print(f"  - total_connections_comparison.png")
    print(f"  - connectivity_density_comparison.png")
    print(f"  - search_time_comparison.png")
    print(f"  - comparison_dashboard.png")
    print(f"  - comparison_summary.txt")

def main():
    parser = argparse.ArgumentParser(description='Generate graphs from HNSW degradation test CSV data')
    parser.add_argument('csv_file', nargs='?', help='Path to the CSV file generated by the degradation test')
    parser.add_argument('-o', '--output', default='graphs', help='Output directory for graphs (default: graphs)')
    parser.add_argument('--list-csv', action='store_true', help='List available CSV files in current directory')
    parser.add_argument('--compare', nargs=2, metavar=('CSV1', 'CSV2'), 
                       help='Compare two CSV files (e.g., --compare without_lsh.csv with_lsh.csv)')
    parser.add_argument('--labels', nargs=2, metavar=('LABEL1', 'LABEL2'), 
                       default=['Without LSH', 'With LSH'],
                       help='Labels for comparison (default: "Without LSH" "With LSH")')
    
    args = parser.parse_args()
    
    if args.list_csv:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'degradation' in f]
        if csv_files:
            print("Available degradation test CSV files:")
            for f in sorted(csv_files):
                print(f"  {f}")
        else:
            print("No degradation test CSV files found in current directory")
        return
    
    if args.compare:
        csv1, csv2 = args.compare
        if not os.path.exists(csv1):
            print(f"Error: CSV file '{csv1}' not found")
            return
        if not os.path.exists(csv2):
            print(f"Error: CSV file '{csv2}' not found")
            return
        
        create_comparison_graphs(csv1, csv2, args.output, args.labels[0], args.labels[1])
        return
    
    if not args.csv_file:
        print("Error: Please provide a CSV file or use --compare option")
        print("Use --help for more information")
        return
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found")
        return
    
    create_graphs(args.csv_file, args.output)

if __name__ == "__main__":
    main()
