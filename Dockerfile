# Use the official Python image from the Amazon ECR Public Gallery as the base image
FROM public.ecr.aws/docker/library/python:3.11

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevent Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensure that the Python output is sent straight to the terminal (e.g., for logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the host to the /app directory in the container
COPY requirements.txt /app/

# Upgrade pip, setuptools, and wheel, then install the dependencies listed in requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the entire contents of the current directory on the host to the /app directory in the container
COPY . /app/

# Copy the pre-trained model file to the /app/model directory in the container
COPY models/iris_model.joblib /app/models/iris_model.joblib

# Expose port 8002 to allow communication to/from the container
EXPOSE 8000

# Define the command to run the application using uvicorn
# This will run the 'app' object from the 'main' module, listening on all network interfaces (0.0.0.0) and port 8000
#chek which port you have assigned in app, otherwise we need to do mapping
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t demo_scale .
#docker run -d -p 8000:8000 demo_scale
#http://localhost:8000
