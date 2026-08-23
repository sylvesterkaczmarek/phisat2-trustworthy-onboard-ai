PYTHON ?= python
EXAMPLE := examples/phi2-eo-tile-filter

.PHONY: install install-reference lint test check demo robustness clean

install:
	$(PYTHON) -m pip install -e "$(EXAMPLE)[dev]"

install-reference:
	$(PYTHON) -m pip install -r $(EXAMPLE)/requirements-reference.txt
	$(PYTHON) -m pip install --no-deps -e $(EXAMPLE)

lint:
	$(PYTHON) -m ruff check --config $(EXAMPLE)/pyproject.toml assurance $(EXAMPLE)

test:
	cd $(EXAMPLE) && $(PYTHON) -m pytest -q

check: lint test

demo:
	cd $(EXAMPLE) && $(PYTHON) scripts/run_demo.py --n 160 --size 32 --bands 3 --epochs 2

robustness:
	cd $(EXAMPLE) && $(PYTHON) scripts/run_robustness_benchmark.py

clean:
	@test "$(EXAMPLE)" = "examples/phi2-eo-tile-filter" || { echo "Refusing unsafe clean target: $(EXAMPLE)" >&2; exit 1; }
	rm -rf $(EXAMPLE)/tiles $(EXAMPLE)/logs $(EXAMPLE)/reports $(EXAMPLE)/runs $(EXAMPLE)/downlink
	rm -rf $(EXAMPLE)/robustness_benchmark $(EXAMPLE)/robustness_downlink
	rm -rf $(EXAMPLE)/models/candidate_bundle $(EXAMPLE)/models/bundles
	rm -f $(EXAMPLE)/models/*.onnx $(EXAMPLE)/models/*.json $(EXAMPLE)/calibration.json
