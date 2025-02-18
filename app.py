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
#import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
import numpy as np
import matplotlib
import wandb

# Directly hardcode the key

# Initialize wandb project
wandb.init(project="billion-row-analysis", name="benchmarking")

# Set Modin to use Dask as the engine
os.environ["MODIN_ENGINE"] = "dask"

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
def load_data_python_vectorized():
    df = pd.read_parquet('data.parquet')
    
    # Convert numerical columns to NumPy arrays for vectorized operations
    num_cols = df.select_dtypes(include=['number']).columns
    np_data = {col: df[col].to_numpy() for col in num_cols}
    return np_data

def load_data_pandas():
    return pd.read_parquet('data.parquet')

def load_data_dask():
    return dd.read_parquet('data.parquet')

def load_data_polars():
    return pl.read_parquet('data.parquet')

# Uncomment to use DuckDB
#def load_data_duckdb():
    #return duckdb.read_parquet('data/raw/jan_2024.parquet')

# Loaders list
loaders = [
    (load_data_pandas, "Pandas"),
    (load_data_dask, "Dask"),
    (load_data_polars, "Polars"),
    # Uncomment for DuckDB
    #(load_data_duckdb, "DuckDB"),
    #(load_data_python_vectorized, "Python Vectorized"),
]

def run_benchmark():
    benchmark_results = []
    error_messages = []
    
    for loader, lib_name in loaders:
        try:
            data, load_time, cpu_load, mem_load, peak_mem_load = measure_performance(loader)

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

def explore_dataset():
    try:
        df = pd.read_parquet('data.parquet')
        
        # Generate dataset summary
        summary = df.describe(include='all').T  
        summary["missing_values"] = df.isnull().sum()
        summary["unique_values"] = df.nunique()
        summary_text = summary.to_markdown()
        
        # Log dataset summary as text in Weights & Biases
        wandb.log({"Dataset Summary": wandb.Html(summary_text)})

        return summary_text, None

    except Exception as e:
        return f"Error loading data: {str(e)}", None

# Gradio interface setup
def gradio_interface():
    def run_and_plot():
        results, plot = run_benchmark()
        return results, plot
    
    def explore_data():
        summary, plot = explore_dataset()
        return summary, plot    

    with gr.Blocks() as demo:
        gr.Markdown("## Explore Dataset")
        explore_button = gr.Button("Explore Data")
        summary_text = gr.Textbox(label="Dataset Summary")
        explore_image = gr.Image(label="Feature Distributions")
        explore_button.click(explore_data, outputs=[summary_text, explore_image])
        
        gr.Markdown("## Benchmarking Different Data Loading Libraries")
        
        run_button = gr.Button("Run Benchmark")
        result_text = gr.Textbox(label="Benchmark Results")
        plot_image = gr.Image(label="Performance Graph")
        
        run_button.click(run_and_plot, outputs=[result_text, plot_image])
    return demo

demo = gradio_interface()

# Run the Gradio app 
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
