from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import time
import psutil
import tracemalloc
import gc
import pandas as pd
import dask.dataframe as dd
import polars as pl
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
import numpy as np
import matplotlib
import wandb
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from datasets import load_dataset

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance measurement function
def measure_performance(load_function, *args):
    gc.collect()
    tracemalloc.start()
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=1)
    
    total_memory = psutil.virtual_memory().total  # Get total system memory
    
    start_memory = psutil.Process().memory_info().rss / total_memory * 100  # Convert to percentage
    data = load_function(*args)
    end_memory = psutil.Process().memory_info().rss / total_memory * 100  # Convert to percentage
    
    end_cpu = psutil.cpu_percent(interval=1)
    end_time = time.time()
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_memory_percentage = peak_memory / total_memory * 100  # Convert to percentage
    
    return data, end_time - start_time, max(end_cpu - start_cpu, 0), max(end_memory - start_memory, 0), peak_memory_percentage

# Data loading functions
def load_data_python_vectorized(df):
    # Convert numerical columns to NumPy arrays for vectorized operations
    num_cols = df.select_dtypes(include=['number']).columns
    np_data = {col: df[col].to_numpy() for col in num_cols}
    return np_data

def load_data_pandas(df):
    return df

def load_data_dask(df):
    return dd.from_pandas(df, npartitions=10)

def load_data_polars(df):
    return pl.from_pandas(df)

def load_data_duckdb(df):
    return duckdb.from_df(df)
    

# Loaders list
loaders = [
    (load_data_pandas, "Pandas"),
    (load_data_dask, "Dask"),
    (load_data_polars, "Polars"),
    (load_data_duckdb, "DuckDB"),
    (load_data_python_vectorized, "Python Vectorized"),
]

def run_benchmark(df):
    benchmark_results = []
    error_messages = []
    
    for loader, lib_name in loaders:
        try:
            data, load_time, cpu_load, mem_load, peak_mem_load = measure_performance(loader, df)

            # Log metrics to Weights & Biases
            wandb.log({
                "Library": lib_name,
                "Load Time (s)": load_time,
                "CPU Load (%)": cpu_load,
                "Memory Load (%)": mem_load,
                "Peak Memory (%)": peak_mem_load
            })

            benchmark_results.append({
                "Library": lib_name,
                "Load Time (s)": load_time,
                "CPU Load (%)": cpu_load,
                "Memory Load (%)": mem_load,
                "Peak Memory (%)": peak_mem_load
            })

        except Exception as e:
            error_messages.append(f"{lib_name} Error: {str(e)}")

    if error_messages:
        return '\n'.join(error_messages), None

    benchmark_df = pd.DataFrame(benchmark_results)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark Results", fontsize=16)

    sns.barplot(x="Library", y="Load Time (s)", data=benchmark_df, ax=axes[0, 0])
    sns.barplot(x="Library", y="CPU Load (%)", data=benchmark_df, ax=axes[0, 1])
    sns.barplot(x="Library", y="Memory Load (%)", data=benchmark_df, ax=axes[1, 0])
    sns.barplot(x="Library", y="Peak Memory (%)", data=benchmark_df, ax=axes[1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert plot to an image and log it to wandb
    image = Image.open(buf)
    wandb.log({"Benchmark Results": wandb.Image(image)})

    image_array = np.array(image)

    return benchmark_df.to_markdown(), image_array  # Return NumPy array

matplotlib.use("Agg")
def explore_dataset(df):
    try:
        # Convert float64 columns to float32 to reduce memory usage
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # If dataset is too large, sample 10%
        if len(df) > 1_000_000:
            df = df.sample(frac=0.5, random_state=42)

        # Generate dataset summary
        summary = df.describe(include='all').T  
        summary["missing_values"] = df.isnull().sum()
        summary["unique_values"] = df.nunique()
        summary_text = summary.to_markdown()
        
        # Log dataset summary as text in Weights & Biases
        wandb.log({"Dataset Summary": wandb.Html(summary_text)})

        # Prepare for visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))  
        fig.suptitle("Dataset Overview", fontsize=16)

        # Plot data type distribution
        data_types = df.dtypes.value_counts()
        sns.barplot(x=data_types.index.astype(str), y=data_types.values, ax=axes[0, 0])
        axes[0, 0].set_title("Column Count by Data Type by Abdul Rafay")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_xlabel("Column Type")

        # Plot mean values of numeric columns
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            mean_values = df[num_cols].mean()
            sns.barplot(x=mean_values.index, y=mean_values.values, ax=axes[0, 1])
            axes[0, 1].set_title("Mean Values of Numeric Columns")
            axes[0, 1].set_xlabel("Column Name")
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Log mean values to Weights & Biases
            for col, mean_val in mean_values.items():
                wandb.log({f"Mean Values/{col}": mean_val})

        # Plot histogram for a selected numerical column
        if len(num_cols) > 0:
            selected_col = num_cols[0]  # Choose the first numeric column
            sns.histplot(df[selected_col], bins=30, kde=True, ax=axes[1, 0])
            axes[1, 0].set_title(f"Distribution of {selected_col}")
            axes[1, 0].set_xlabel(selected_col)
            axes[1, 0].set_ylabel("Frequency")

        # Plot correlation heatmap
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[1, 1])
            axes[1, 1].set_title("Correlation Heatmap")

        # Save figure to buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Convert figure to NumPy array
        image = Image.open(buf)
        image_array = np.array(image)

        # Log image to Weights & Biases
        wandb.log({"Dataset Overview": wandb.Image(image)})

        return summary_text, image_array

    except Exception as e:
        return f"Error loading data: {str(e)}", None

# New functionality: Group By, Filtering, Pure Python Loop, Multiprocessing
def group_by_column(df, column):
    return df.groupby(column).size().reset_index(name='count')

def filter_data(df, column, condition, value):
    if condition == ">":
        return df[df[column] > value]
    elif condition == "<":
        return df[df[column] < value]
    elif condition == "==":
        return df[df[column] == value]
    elif condition == "!=":
        return df[df[column] != value]
    else:
        return df

def pure_python_loop(df, column):
    result = {}
    for value in df[column]:
        result[value] = result.get(value, 0) + 1
    return result

def multiprocessing_loop(df, column):
    def process_chunk(chunk):
        result = {}
        for value in chunk:
            result[value] = result.get(value, 0) + 1
        return result

    num_cores = cpu_count()
    chunk_size = len(df) // num_cores
    chunks = [df[column].iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    final_result = {}
    for result in results:
        for key, value in result.items():
            final_result[key] = final_result.get(key, 0) + value
    return final_result



# Dataset options from Hugging Face
DATASET_OPTIONS = {
    "NYC Taxi Data": "iampalina/nyc_taxi",
    "IMDB Reviews": "imdb",
    "Amazon Reviews": "amazon_polarity",
    "Hate Speech": "hate_speech18",
    "Titanic Dataset": "Kaggle/titanic",
}

# Function to load dataset
def load_selected_dataset(dataset_key):
    try:
        dataset_name = DATASET_OPTIONS[dataset_key]
        dataset = load_dataset(dataset_name, split="train")
        return dataset.to_pandas()
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

# Function to explore dataset
def explore_data(df):
    if isinstance(df, str):  # Handle error messages
        return df, None
    summary, plot = explore_dataset(df)
    return summary, plot    

# Function to process data
def process_data(df, operation, column, condition=None, value=None):
    if isinstance(df, str):  # Handle error messages
        return df
    if operation == "Group By":
        return str(group_by_column(df, column))
    elif operation == "Filter":
        return str(filter_data(df, column, condition, value))
    elif operation == "Pure Python Loop":
        return str(pure_python_loop(df, column))
    elif operation == "Multiprocessing Loop":
        return str(multiprocessing_loop(df, column))
    return "Invalid operation selected."

# Function to run and plot benchmark
def run_and_plot(df):
    if isinstance(df, str):  # Handle error messages
        return df, None
    results, plot = run_benchmark(df)
    return results, plot

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Select a Dataset from Hugging Face Hub")

        dataset_dropdown = gr.Dropdown(choices=list(DATASET_OPTIONS.keys()), label="Select Dataset")
        load_button = gr.Button("Load Dataset")
        df_state = gr.State(None)  # ✅ Initialize df_state properly

        # Load Dataset
        summary_text = gr.Textbox(label="Dataset Summary")
        explore_image = gr.Image(label="Feature Distributions")

        def update_dataset(selected_dataset):
            df = load_selected_dataset(selected_dataset)
            df_state.value = df  # ✅ Store dataset in state
            return f"Dataset '{selected_dataset}' loaded successfully."

        load_button.click(update_dataset, inputs=dataset_dropdown, outputs=summary_text)

        # Explore Dataset
        gr.Markdown("## Explore Dataset")
        explore_button = gr.Button("Explore Data")
        explore_button.click(explore_data, inputs=df_state, outputs=[summary_text, explore_image])

        # Data Processing
        gr.Markdown("## Data Processing")
        operation = gr.Dropdown(["Group By", "Filter", "Pure Python Loop", "Multiprocessing Loop"], label="Operation")
        column = gr.Textbox(label="Column Name")
        condition = gr.Dropdown([">", "<", "==", "!="], label="Condition (for Filter)", interactive=True)
        value = gr.Number(label="Value (for Filter)", interactive=True)
        process_button = gr.Button("Process Data")
        result_text = gr.Textbox(label="Processing Result")
        process_button.click(process_data, inputs=[df_state, operation, column, condition, value], outputs=result_text)

        # Benchmarking
        gr.Markdown("## Benchmarking Different Data Loading Libraries")
        run_button = gr.Button("Run Benchmark")
        result_text_benchmark = gr.Textbox(label="Benchmark Results")
        plot_image = gr.Image(label="Performance Graph")
        run_button.click(run_and_plot, inputs=df_state, outputs=[result_text_benchmark, plot_image])

    return demo


# Initialize W&B
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="billion-row-analysis", name="benchmarking")

# Run the Gradio app



demo = gradio_interface()
demo.launch(share=False)  # No need for share=True in VS Code, local access is sufficient