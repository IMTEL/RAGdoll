# RAGdoll

# 🚀 RAGdoll development tool

available api at https://iplvr.it.ntnu.no

## 📦 Installation
### Prerequisites
- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Docker is used for the backend and database setup. [Download Docker](https://www.docker.com/products/docker-desktop)

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/IMTEL/chat-service.git
cd chat-service
```

### **2️⃣ Set Up a Virtual Environment (Recommended)**
It’s best to install dependencies inside a virtual environment:

**1. Create a Virtual Environment**


Run the following command to create a virtual environment named venv:
<pre> python -m venv venv </pre>


**2. Activate the Virtual Environment**

Before installing or updating packages, activate the virtual environment.

*On windows:*
<pre>source venv/Scripts/activate</pre>

*On linux and mac:*
<pre>source venv/bin/activate</pre>



**3. Install Existing Dependencies**


follow [manual for updating requirements](docs/manuals/update_requirements.md)

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the root directory of the project and add the following environment variables:
# 🚀 Running the Service Locally

### **1️⃣ Start the FastAPI Server**
Run the following command:

```sh
uvicorn src.main:app --reload
```
The service will now be available at:

Docs UI: http://127.0.0.1:8000/docs
API Root: http://127.0.0.1:8000

### **2️⃣ Verify the /ping Endpoint**
Check if the service is running:
The `/ping` endpoint is used to check if the backend is running.

```sh
curl http://127.0.0.1:8000/ping
```
✅ Expected response:
```sh
{"status":"I AM ALIVE!"}
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

### 3 Verify the Service
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
### Configuration

Create a `.env` file in the root directory by following the .env.example

# Testing
without docker:
```bash
pytest --cov=src --cov-report=term-missing
```
Docker:
```bash
docker compose build
docker compose up -d
docker compose run chat-service pytest
docker compose down
```
##Test endpoint with mock data
### curl command to test the endpoint:

curl -X POST "http://localhost:8000/api/progress" \
-H "Content-Type: application/json" \
-d '{
  "task_name": "Daily Exercise Routine",
  "status": "start",
  "user_id": "user123",
  "subtask_progress": [
    {
      "subtask_name": "Warm Up",
      "description": "Prepare muscles for workout",
      "completed": false,
      "step_progress": [
        {
          "step_name": "Jumping Jacks",
          "repetition_number": 30,
          "completed": false
        },
        {
          "step_name": "Arm Circles",
          "repetition_number": 20,
          "completed": false
        }
      ]
    },
    {
      "subtask_name": "Main Workout",
      "description": "Intense exercise session",
      "completed": false,
      "step_progress": [
        {
          "step_name": "Push Ups",
          "repetition_number": 50,
          "completed": false
        }
      ]
    }
  ]
}

curl -X POST "http://localhost:8000/api/progress" \
-H "Content-Type: application/json" \
-d '{
  "task_name": "Daily Exercise Routine",
  "status": "complete",
  "user_id": "user123",
  "subtask_progress": [
    {
      "subtask_name": "Warm Up",
      "description": "Prepare muscles for workout",
      "completed": true,
      "step_progress": [
        {
          "step_name": "Jumping Jacks",
          "repetition_number": 30,
          "completed": true
        },
        {
          "step_name": "Arm Circles",
          "repetition_number": 20,
          "completed": true
        }
      ]
    },
    {
      "subtask_name": "Main Workout",
      "description": "Intense exercise session",
      "completed": true,
      "step_progress": [
        {
          "step_name": "Push Ups",
          "repetition_number": 50,
          "completed": true
        }
      ]
    }
  ]
}'

### Receive the log
curl -X GET "http://localhost:8000/api/progress"



## Contributors

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/tobiasfremming">
          <img src="https://github.com/tobiasfremming.png?size=100" width="100px;"/><br />
          <sub><b>Tobias Fremming</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/haluboi">
          <img src="https://github.com/haluboi.png?size=100" width="100px;"/><br />
          <sub><b>Halvor Heien Førde</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/peter-olai">
          <img src="https://github.com/peter-olai.png?size=100" width="100px;"/><br />
          <sub><b>Peter Olai Johnsen</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hallvardvo">
          <img src="https://github.com/hallvardvo.png?size=100" width="100px;"/><br />
          <sub><b>Hallvard Vatnar Olsen</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/erikleblanc">
          <img src="https://github.com/erikleblanc.png?size=100" width="100px;"/><br />
          <sub><b>Erik Le Blanc Pleym</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/selinyo">
          <img src="https://github.com/selinyo.png?size=100" width="100px;"/><br />
          <sub><b>Selin Yuki Øzkan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/snobohle">
          <img src="https://github.com/snobohle.png?size=100" width="100px;"/><br />
          <sub><b>Erik Olsen Bøhle</b></sub>
      </a>
    </td>
  </tr>
</table>
