import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations(df, output_dir):
    """Generates the mandatory EDA visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating data visualizations...")
    
    # Set seaborn style for better visuals
    sns.set_theme(style="whitegrid")
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # 1. Distribution/histogram plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Distribution of Numerical Features', fontsize=16)
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i//2, i%2], color='skyblue')
        axes[i//2, i%2].set_title(f'Histogram of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_histograms.png'))
    plt.close()
    
    # 2. Boxplots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[numeric_cols], orient='h', palette='Set2')
    plt.title('Boxplots of Numerical Features')
    plt.savefig(os.path.join(output_dir, '2_boxplots.png'))
    plt.close()
    
    # 3. Pairplot
    pairplot = sns.pairplot(df, hue='species', palette='Dark2', markers=["o", "s", "D"])
    pairplot.fig.suptitle('Pairplot of Features by Species', y=1.02)
    pairplot.savefig(os.path.join(output_dir, '3_pairplot.png'))
    plt.close('all')
    
    # 4. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, '4_correlation_heatmap.png'))
    plt.close()
    
    # 5. Class distribution chart
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='species', hue='species', palette='pastel', legend=False)
    plt.title('Target Class Distribution')
    plt.ylabel('Count')
    plt.xlabel('Species')
    plt.savefig(os.path.join(output_dir, '5_class_distribution.png'))
    plt.close()
    
    # 6. Scatter plot of important features (Petal Length vs Width)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='deep', s=100)
    plt.title('Petal Length vs Petal Width (Key Separators)')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend(title='Species')
    plt.savefig(os.path.join(output_dir, '6_scatter_plot.png'))
    plt.close()
    
    print(f"Visualizations saved successfully to {output_dir}")
