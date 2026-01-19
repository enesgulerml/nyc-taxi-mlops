# ESTABLISH THE VIRTUAL ENVIRONMENT AND ITS ADDICTIONS.
install:
	pip install -r requirements.txt

# TRAIN THE MODEL LOCALLY (WITHOUT DOCKER).
train:
	python -m src.pipelines.training_pipeline

# SET UP THE API AND UI WITH DOCKER (INCLUDING THE BUILD).
up:
	docker-compose up --build

# STOP AND DELETE CONTAINER
down:
	docker-compose down

# FOR API TESTING ONLY
test:
	pytest

# CLEAN DOCKER IMAGES AND CACHE (DEEP CLEAN)
clean:
	docker system prune -f