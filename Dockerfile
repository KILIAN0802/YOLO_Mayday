# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV (libgl1-mesa-glx and libglib2.0-0)
# These are crucial for cv2 to work correctly in a headless environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for model weights
RUN mkdir -p /app/weights

# Copy the trained model weights from your local 'weights' folder.
# Place best.pt in the repository's weights/ directory before building the image.
COPY weights/best.pt /app/weights/best.pt

# Copy the inference core and Flask application files
COPY inference_core.py .
COPY app.py .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]