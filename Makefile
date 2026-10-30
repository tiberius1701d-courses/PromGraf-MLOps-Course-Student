.PHONY: all stop evaluation

all:
	docker compose up --build -d

stop:
	docker compose down

evaluation:
	docker compose up --build -d evaluation
