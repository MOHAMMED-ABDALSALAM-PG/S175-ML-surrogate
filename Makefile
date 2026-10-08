PY ?= $(HOME)/env314/bin/python
SEED ?= 0
SPLIT ?= S1_random
CONFIG ?= configs/v3_masked_peroutput.yaml
RAW ?= $(HOME)/S175_shaft_gen_off.txt

.PHONY: test prepare pilot train train-all timing clean
test:      ; $(PY) tests/test_core.py
prepare:   ; $(PY) scripts/01_prepare.py --raw $(RAW)
pilot:     ; $(PY) scripts/02_train.py --config $(CONFIG) --pilot
train:     ; $(PY) scripts/02_train.py --config $(CONFIG) --split $(SPLIT) --seed $(SEED)
train-all: ; bash scripts/run_all.sh
timing:    ; $(PY) scripts/06_timing_simulator.py
clean:     ; rm -rf data/cache

# smoke / evaluate / figures targets return once scripts/00_smoke.py,
# 04_evaluate.py and 05_figures.py are written -- see the README roadmap.
