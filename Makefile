.PHONY: install test demo clean

install:
	python -m pip install -e "examples/phi2-eo-tile-filter[dev]"

test:
	cd examples/phi2-eo-tile-filter && python -m pytest -q

demo:
	cd examples/phi2-eo-tile-filter && python scripts/run_demo.py --n 160 --size 32 --bands 3 --epochs 2

clean:
	rm -rf examples/phi2-eo-tile-filter/{tiles,logs,reports,runs,downlink}
	rm -f examples/phi2-eo-tile-filter/models/*.onnx examples/phi2-eo-tile-filter/models/*.json
