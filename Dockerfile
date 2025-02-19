# Use an official Python runtime as the base image
FROM python:3.13.2

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY   FAST_APPLICATION/app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI and Gradio ports
EXPOSE 8000 7860

# Define the environment variable for W&B API key
ENV WANDB_API_KEY="your-wandb-api-key-here"

# Run the FastAPI and Gradio app
CMD ["python", "app.py"]
