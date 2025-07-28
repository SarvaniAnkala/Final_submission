# Use an official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create input/output dirs if not mounted
RUN mkdir -p /app/input /app/output

# Run the script when the container starts
CMD ["python", "app.py"]
