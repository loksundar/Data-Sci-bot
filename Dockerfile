
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
# This is the default port Streamlit runs on
EXPOSE 8501

# Define environment variable for the port (Cloud Run sets this)
ENV PORT 8501

# Run app.py when the container launches
# Use environment variable $PORT for compatibility with Cloud Run
# --server.enableCORS false and --server.enableXsrfProtection false can help with some proxy/embedding scenarios
CMD ["streamlit", "run", "app.py", "--server.port", "$PORT", "--server.address", "0.0.0.0", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
