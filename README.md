# 🚖 NYC Taxi Duration Prediction (End-to-End MLOps)
An end-to-end MLOps project that predicts taxi trip duration in New York City.
This project demonstrates the transition from a research notebook to **a production-grade microservices architecture**, featuring optimized Docker builds, secure Kubernetes deployment, and high-performance inference using ONNX Runtime.

---

## 🏗 Architecture
The system follows a decoupled **Microservices Architecture:**
* **Frontend Service:** Interactive UI built with Streamlit.
* **Backend Service:** High-performance REST API built with FastAPI.
* **Inference Engine:** ONNX Runtime used instead of Scikit-learn for faster inference and smaller image size.
* **Caching Layer:** Redis to cache frequent prediction requests (latency reduction).
* **Orchestration:** Fully containerized with Docker and deployed on Kubernetes (Minikube).

---
## 🚀 Key Technical Highlights
### 1. 📉 Docker Image Optimization (2.05GB → 650MB)
**Problem:** The initial monolithic image was ~2GB due to heavy libraries (Scikit-learn, XGBoost, Pandas) used for training.
* **Solution:**
  * **Decoupling:** Separated API and UI into distinct Docker images.
  * **Dependency Pruning:** Removed heavy training libraries (xgboost, sklearn) from the production API image. Only onnxruntime and pandas are kept for inference.
  * **Multi-Stage Builds:** Used Docker multi-stage builds to exclude build tools and cache files from the final runtime image.
  * **Context Optimization:** Implemented strict .dockerignore rules (excluding venv, models, git).

### 2. 🛡️ Kubernetes Security & Hardening
* **Non-Root User:** Containers run as a dedicated non-root user (appuser UID: 1000) instead of default root, following security best practices.
* **Security Context:** Configured Kubernetes securityContext (fsGroup: 1000) to manage volume permissions securely, preventing "Permission Denied" errors on log volumes.
* **Volume Management:** Used emptyDir volumes for temporary log storage in a read-only container environment.

### 3. ⚡ Performance Engineering
* **ONNX Format:** Trained Random Forest model is converted to .onnx format, reducing the dependency footprint and improving inference speed.
* **Redis Caching:** Implemented a caching mechanism in FastAPI. Identical requests are served from Redis (Memory) instead of re-running the model.
* **Lifespan Events:** Model loading and Redis connection happen only once during application startup (using FastAPI lifespan), preventing I/O overhead on every request.

---
## 📂 Project Structure

```text
.
├── k8s/                     # Kubernetes Deployment & Service Manifests
│   ├── api.yaml             # API Deployment (Security Context + Volumes)
│   ├── ui.yaml              # UI Deployment
│   └── redis.yaml           # Redis Service
├── src/
│   ├── api/                 # Backend Microservice
│   │   ├── Dockerfile       # Multi-stage optimized Dockerfile
│   │   ├── main.py          # FastAPI application (w/ Redis & ONNX)
│   │   └── requirements.txt # Minimal production dependencies
│   ├── frontend/            # Frontend Microservice
│   │   ├── Dockerfile       # Multi-stage optimized Dockerfile
│   │   └── ui.py            # Streamlit Dashboard
│   │   └── requirements.txt # Minimal production dependencies
│   ├── pipelines/           # Training Pipelines (Offline)
│   └── utils/               # Shared utilities (Logger, Config)
│   └── components/          # Reusable ML modules (Feature Eng., Processing) 
├── tests/                   # Unit & Integration tests
├── docker-compose.yml       # Local development orchestration
├── Makefile                 # Automation shortcuts
└── README.md                # Project Documentation
```

---
## 🛠️ Installation & Usage
### Option 1: Docker Compose (Recommended for Testing)
Run the entire stack (API + UI + Redis) with a single command:
```bash
    # Build and Start Services
    make docker-up
    
    # Check Logs
    make docker-logs
    
    # Stop Services
    make docker-down
```
* **UI:** http://localhost:8501
* **API Docs:** http://localhost:8000/docs

### Option 2: Kubernetes (Minikube)
Deploy to a local Kubernetes cluster:

```bash
    # 1. Start Minikube
    minikube start
    
    # 2. Load Images into Minikube (Important for local images)
    minikube image load nyc-taxi-api_service:latest
    minikube image load nyc-taxi-ui_service:latest
    
    # 3. Apply Manifests
    kubectl apply -f k8s/redis.yaml
    kubectl apply -f k8s/api.yaml
    kubectl apply -f k8s/ui.yaml
    
    # 4. Access the UI
    minikube service ui-service
```

---
## 🧪 Training Pipeline (Offline)
The training pipeline (src/pipelines/training_pipeline.py) is decoupled from the production API.
1. Loads raw data.
2. Performs Feature Engineering.
3. Trains a Random Forest model.
4. Logs experiments and metrics to MLflow.
5. Converts and saves the best model as ONNX (model.onnx).

---
## 📊 Results & Metrics
* **Model:** HistGradientBoosting Regressor (ONNX)
* **Inference Latency:** ~15ms (P95)
* **Docker Image Size:** ~650MB (Reduced from 2GB)
* **Security Compliance:** Non-root execution

---
## 👨‍💻 Author
Enes Guler - Junior MLOps Engineer