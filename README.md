# Geometry-Aware Triplane Diffusion for Single Shape Generation with Feature

## Install required packages

```bash
conda create -n gatd python=3.10
conda activate gatd 
pip install -r requirements.txt
```

## Quick test

```bash
bash run.sh
```

### Data

```bash
cd data
python mesh_sampler.py -s {OBJ_PATH} -d {SAVE_NPZ_PATH} --n_surf 5000000
```

### Training

```bash
cd src
python train.py --tag {EXP_DIR} --data_path {NPZ_PATH} --enable_multiscale True --gpu_id 0
```

### Sampling

```bash
cd src
python sample.py --tag {EXP_DIR} --n_samples 10 --n_faces 50000 --output results10 --gpu_id 0
```

### Evaluation

```bash
cd rendering
python mvrender_script.py -s {RESULT_DIR} -g 0 -bl {BLENDER_PATH}
cd evaluation
python eval_full.py -s {RESULT_DIR} -r {NPZ_DATA_DIR} -g 0
```
