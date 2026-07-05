import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Custom professional color palette
COLORS_COMPARE = ['#4A90E2', '#50E3C2', '#F5A623']
COLORS_TWO = ['#5C6BC0', '#26A69A']

def plot_model_size():
    formats = ['FP32 Baseline', 'Dynamic Range (DRQ)', 'Strict INT8']
    sizes = [24.92, 6.09, 6.49]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(formats, sizes, color=COLORS_COMPARE, height=0.55, edgecolor='black', linewidth=0.7)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f} MB', 
                ha='left', va='center', fontweight='bold', color='#333333')
        
    ax.set_xlim(0, 30)
    ax.set_xlabel('Model Size (MB)', fontweight='bold')
    ax.set_title('YOLOv7-Tiny Model Size Comparison', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/model_size_comparison.png', dpi=300)
    plt.close()
    print("Saved plots/model_size_comparison.png")

def plot_latency():
    categories = ['Initialization', 'Warmup Average', 'Steady Inference']
    fp32_latency = [202.12, 37.04, 35.20]
    int8_latency = [91.98, 31.70, 31.31]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, fp32_latency, width, label='FP32 TFLite', color=COLORS_TWO[0], edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x + width/2, int8_latency, width, label='Strict INT8 TFLite', color=COLORS_TWO[1], edgecolor='black', linewidth=0.7)
    
    # Label height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_ylabel('Latency (milliseconds)', fontweight='bold')
    ax.set_title('WSL2 CPU Execution Latency (4 Threads)', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend(frameon=True)
    ax.set_ylim(0, 240)
    plt.tight_layout()
    plt.savefig('plots/latency_comparison.png', dpi=300)
    plt.close()
    print("Saved plots/latency_comparison.png")

def plot_memory():
    categories = ['Initialization Peak', 'Overall Run Peak']
    fp32_mem = [91.17, 112.92]
    int8_mem = [31.05, 36.55]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, fp32_mem, width, label='FP32 TFLite', color=COLORS_TWO[0], edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x + width/2, int8_mem, width, label='Strict INT8 TFLite', color=COLORS_TWO[1], edgecolor='black', linewidth=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f} MB',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_ylabel('Peak Memory Consumption (MB)', fontweight='bold')
    ax.set_title('Benchmark Memory Footprint Delta', pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend(frameon=True)
    ax.set_ylim(0, 135)
    plt.tight_layout()
    plt.savefig('plots/memory_comparison.png', dpi=300)
    plt.close()
    print("Saved plots/memory_comparison.png")

def plot_accuracy():
    formats = ['Dynamic Range (DRQ)', 'Strict INT8']
    match_rates = [92.54, 88.06]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(formats, match_rates, color=['#7E57C2', '#26A69A'], width=0.45, edgecolor='black', linewidth=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    ax.set_ylabel('Match Rate against PyTorch (%)', fontweight='bold')
    ax.set_title('Empirical Accuracy Match Rate (COCO-128 Subset)', pad=15, fontweight='bold')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig('plots/accuracy_comparison.png', dpi=300)
    plt.close()
    print("Saved plots/accuracy_comparison.png")

if __name__ == "__main__":
    plot_model_size()
    plot_latency()
    plot_memory()
    plot_accuracy()
    print("All charts generated successfully.")
