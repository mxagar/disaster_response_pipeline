# We can modify image/python version with
# docker build --build-arg IMAGE=python:3.8
# Otherwise, default: python:3.9.16
ARG IMAGE=python:3.9.16
FROM $IMAGE

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

# Create directory IN container and change to it
WORKDIR /opt/disaster_response_pipeline

# Copy folder contents (unless the ones from .dockerignore) TO container
ADD . /opt/disaster_response_pipeline/
# Install requirements
RUN pip install --upgrade pip
RUN pip install -r /opt/disaster_response_pipeline/requirements.txt --no-cache-dir
RUN pip install .

# Change permissions
RUN chmod +x /opt/disaster_response_pipeline/run.sh
RUN chown -R ml-api-user:ml-api-user ./

# Change user to the one created
USER ml-api-user

# Expose port
EXPOSE 3000

# Run web server, started by run.sh
# pyhon app/run.py
CMD ["bash", "./run.sh"]

# Build the Dockerfile to create the image
# docker build -t <image_name[:version]> <path/to/Dockerfile>
#   docker build -t disaster_response_app:latest .
# 
# Check the image is there: watch the size (e.g., ~1GB)
#   docker image ls
#
# Run the container locally from a built image
# Recall to: forward ports (-p) and pass PORT env variable (-e)
# Optional: 
# -d to detach/get the shell back,
# --name if we want to choose conatiner name (else, one randomly chosen)
# --rm: automatically remove container after finishing (irrelevant in our case, but...)
#   docker run -d --rm -p 3000:3000 -e PORT=3000 --name disaster_response_app disaster_response_app:latest
#
# Check the App locally: open the browser
#   http://localhost:3000
#   Use the web app
# 
# Check the running containers: check the name/id of our container,
# e.g., disaster_response_app
#   docker container ls
#   docker ps
#
# Get a terminal into the container: in general, BAD practice
# docker exec -it <id|name> sh
#   docker exec -it disaster_response_app sh
#   (we get inside)
#   cd /opt/disaster_response_pipeline
#   ls
#   cat disaster_response_pipeline.log
#   exit
#
# Stop container and remove it (erase all files in it, etc.)
# docker stop <id/name>
# docker rm <id/name>
#   docker stop disaster_response_app
#   docker rm disaster_response_app