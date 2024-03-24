# Use Python as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to container
COPY . .

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
