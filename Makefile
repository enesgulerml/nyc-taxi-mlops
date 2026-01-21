# ==============================================================================
#  NYC TAXI MLOPS PROJECT
# ==============================================================================

PYTHON = python
PIP = pip
DOCKER_COMPOSE = docker-compose
KUBECTL = kubectl
MINIKUBE = minikube
K8S_DIR = k8s

.PHONY: help install train test clean docker-up docker-down k8s-start k8s-build k8s-up k8s-down

# ==============================================================================
#  COMMANDS
# ==============================================================================

help:
	@echo ---------------------------------------------------
	@echo  NYC TAXI MLOPS PROJECT - COMMAND CENTER
	@echo ---------------------------------------------------
	@echo  [ LOCAL DEVELOPMENT ]
	@echo  make install       : Install Python dependencies
	@echo  make train         : Train the model locally
	@echo  make test          : Run unit tests
	@echo ---------------------------------------------------
	@echo  [ DOCKER COMPOSE ]
	@echo  make docker-build  : Build Docker images
	@echo  make docker-up     : Start API and UI
	@echo  make docker-down   : Stop containers
	@echo  make docker-logs   : Shows Logs
	@echo ---------------------------------------------------
	@echo  [ KUBERNETES / MINIKUBE ]
	@echo  make k8s-start     : Start Minikube
	@echo  make k8s-build     : Build images INSIDE Minikube
	@echo  make k8s-up        : Deploy to K8s
	@echo  make k8s-down      : Stops K8s
	@echo  make k8s-restart   : Restart Pods
	@echo  make k8s-forward   : Port-Forward UI
	@echo ---------------------------------------------------
	@echo  [ CLEANING ]
	@echo  make clean     : Deletes Docker
	@echo ---------------------------------------------------

# ==============================================================================
#  LOCAL DEPLOYMENT & TESTS
# ==============================================================================

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) -m src.pipelines.training_pipeline

test:
	pytest

# ==============================================================================
#  DOCKER
# ==============================================================================

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up --build -d
	@echo UI is ready at http://localhost:8501

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

# ==============================================================================
#  KUBERNETES
# ==============================================================================

k8s-start:
	$(MINIKUBE) start --driver=docker
	$(MINIKUBE) addons enable metrics-server

k8s-build:
	@echo Switching to Minikube Env...
	powershell -Command "& minikube -p minikube docker-env --shell powershell | Invoke-Expression; docker-compose build"

k8s-up:
	$(KUBECTL) apply -f $(K8S_DIR)/

k8s-down:
	$(KUBECTL) delete -f $(K8S_DIR)/

k8s-restart:
	$(KUBECTL) rollout restart deployment/api-deployment
	$(KUBECTL) rollout restart deployment/ui-deployment

k8s-forward:
	@echo Forwarding 8501...
	$(KUBECTL) port-forward service/ui-service 8501:8501

# ==============================================================================
#  CLEAN
# ==============================================================================

clean:
	docker system prune -f