import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file as a pandas DataFrame
df = pd.read_json('scores_fixed.json')

# Skip 'prompt' and 'output' columns as requested
df = df.drop(['prompt', 'output'], axis=1)

# Create a pair column for (model, thinking)
df['model_thinking'] = df['model'] + '_' + df['thinking'].astype(str)

# Extract model size
df['model_size'] = df['model'].str.split(':').str[1].str.replace('b', '').astype(float)

# Get unique tasks sorted by task_id
task_mapping = df[['task_id', 'task_name']].drop_duplicates().sort_values('task_id')
unique_tasks = task_mapping['task_name'].tolist()

# Generate barchart for each task: methods (prompt_type) grouped by (model, thinking) pair
for task in unique_tasks:
    task_df = df[df['task_name'] == task]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), gridspec_kw={'width_ratios': [2, 1, 1]})

    # Left subplot: existing barchart
    sns.barplot(data=task_df, x='model_thinking', y='score', hue='prompt_type', ax=ax1, ci=None)
    ax1.set_title(f'Barchart for {task} - Methods Grouped by (Model, Thinking) Pair')
    ax1.set_xlabel('(Model, Thinking)')
    ax1.set_ylabel('Score (%)')
    ax1.legend(title='Prompt Type')
    ax1.tick_params(axis='x', rotation=45)

    # Second subplot: average performance per model
    task_avg = task_df.groupby('model')['score'].mean().reset_index()
    sns.barplot(data=task_avg, x='model', y='score', ax=ax2, ci=None)
    ax2.set_title(f'Average Performance by Model')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Score (%)')

    # Third subplot: average performance by prompting method
    task_avg_method = task_df.groupby('prompt_type')['score'].mean().reset_index()
    sns.barplot(data=task_avg_method, x='prompt_type', y='score', ax=ax3, ci=None)
    ax3.set_title(f'Average Performance by Prompting Method')
    ax3.set_xlabel('Prompt Type')
    ax3.set_ylabel('Average Score (%)')

    plt.tight_layout()
    plt.savefig(f'charts/{task.replace(" ", "_")}_barchart.png')
    plt.close()

# Generate heatmap of task vs prompt technique (average score)
task_prompt_pivot = df.pivot_table(values='score', index='prompt_type', columns='task_name', aggfunc='mean')
task_prompt_pivot = task_prompt_pivot[unique_tasks]
overall_scores = task_prompt_pivot.mean(axis=1).to_frame('Overall')

# Compute global min and max for consistent color scaling
all_data = pd.concat([task_prompt_pivot, overall_scores], axis=1)
min_val = all_data.min().min()
max_val = all_data.max().max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [len(task_prompt_pivot.columns), 1], 'wspace': 0.1})

# Shared colorbar
cbar_ax = fig.add_axes([0.95, 0.2, 0.02, 0.7])

# Heatmap for tasks
sns.heatmap(task_prompt_pivot, annot=True, cmap='viridis', fmt='.1f', ax=ax1, cbar_ax=cbar_ax, vmin=min_val, vmax=max_val)
ax1.set_title('Heatmap of Prompt Technique vs Task')
ax1.set_xlabel('Task')
ax1.set_ylabel('Prompt Technique')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.tick_params(axis='x', labelsize=10)

# Heatmap for overall
sns.heatmap(overall_scores, annot=True, cmap='viridis', fmt='.1f', ax=ax2, cbar=False, vmin=min_val, vmax=max_val)
ax2.set_title('Overall Performance')
ax2.set_xlabel('')
ax2.set_ylabel('')

plt.subplots_adjust(left=0.05, right=0.92, bottom=0.2, top=0.9)
plt.savefig('charts/task_vs_prompt_technique_heatmap.png')
plt.close()

# Generate barchart of method performance (average score per method across all tasks)
method_performance = df.groupby('prompt_type')['score'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=method_performance, x='prompt_type', y='score', ci=None)
plt.title('Barchart of Method Performance')
plt.xlabel('Method (Prompt Type)')
plt.ylabel('Average Score (%)')
plt.savefig('charts/method_performance_barchart.png')
plt.close()

# Generate overall chart for model size vs average score across all tasks
overall_size_avg = df.groupby('model_size')['score'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=overall_size_avg, x='model_size', y='score', ci=None)
plt.title('Model Size vs Average Score Across All Tasks')
plt.xlabel('Model Size (B)')
plt.ylabel('Average Score (%)')
plt.savefig('charts/model_size_vs_score_barchart.png')
plt.close()

# Generate barchart of average score per prompting technique grouped by model-thinking
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='prompt_type', y='score', hue='model_thinking', ci=None)
plt.title('Average Score per Prompting Technique Grouped by Model-Thinking')
plt.xlabel('Prompting Technique')
plt.ylabel('Average Score (%)')
plt.legend(title='Model-Thinking', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('charts/prompt_technique_vs_model_thinking_barchart.png', bbox_inches='tight')
plt.close()

print("Charts generated and saved as PNG files.")
