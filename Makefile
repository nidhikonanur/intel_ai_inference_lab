PYTHON ?= python3

.PHONY: install test benchmark

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

benchmark:
	$(PYTHON) -m inference_lab benchmark --model models/model.onnx --images sample_images --runtime both --iterations 25 --warmup 5
