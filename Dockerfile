# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the required files and directory into the container at /app
COPY service.py /app/service.py
RUN mkdir -p /app/model  # Create the "model" directory before copying
COPY model.joblib /app/model/model.joblib
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

# Run service.py when the container launches
CMD ["python", "service.py"]
