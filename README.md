# chat-service


## Health Check: /ping Endpoint

The `/ping` endpoint is used to check if the backend is running.

### **How to Test `/ping` Using cURL**
Run:
```sh
curl -v http://127.0.0.1:8000/ping
```


##  Running the Service with Docker

This service can be containerized using **Docker** for easy deployment.

### 1️ Build the Docker Image
Run the following command to build the Docker image:

```sh
docker build -t chat-service .
```

### 2 Run the Docker Container
Once the image is built, start the container:

```sh
docker run -p 8000:8000 chat-service
```
The service should now be running at http://127.0.0.1:8000.

### Verify the Service
Test if the service is running by making a request to the /ping endpoint:

# Using cURL:
```sh
curl http://127.0.0.1:8000/ping
```
Expected Response:
{"status":"I AM ALIVE!"}

### 4 Stop the Running Container
To stop the running container:
```sh
docker ps  # Get container ID
docker stop <container_id>
```

### 5 Debugging (Optional)
If the service is not running as expected:

Check running containers:
```sh
docker ps
```
View logs:
```sh
docker logs <container_id>
```
# 📌 Notes
Make sure Docker is installed and running before executing these commands.
The Dockerfile is designed to expose port 8000, so ensure no other service is using this port.