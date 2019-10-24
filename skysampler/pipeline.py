"""
Manages wrapping all parts into a simple pipeline

Key Aspects

1) Identify the task to be done from config file.

2) Define the steps needed to accomplish that

3)







"""

template = """#!/usr/bin/env bash

export OMP_NUM_THREADS=16
export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS

export TMPDIR=/tmp/
#export TMPDIR=/e/eser2/vargatn/DES/SIM_DATA/TMPDIR/
export IMSIM_DIR=/home/moon/vargatn/DES/Y3_WORK/y3-wl_image_sims/
export DES_TEST_DATA=/e/eser2/vargatn/DES/SIM_DATA/DES_TEST_DATA/
export SIM_OUTPUT_DIR=/e/eser2/vargatn/DES/SIM_DATA/
export NGMIX_CONFIG_DIR=${IMSIM_DIR}/ngmix_config
export PIFF_DATA_DIR=/e/eser2/vargatn/DES/SIM_DATA/DES_TEST_DATA/
export PIFF_RUN=y3a1-v29

tag=TAG
version=VERSION
cores_per_job=16

config_file=/home/moon/vargatn/DES/Y3_WORK/skysampler-config/image_sims_configs/CONFIG_FILE
outdir=${SIM_OUTPUT_DIR}/output-${version}-${tag}

python=/home/moon/vargatn/anaconda3/envs/sim2/bin/python
bin=/home/moon/vargatn/DES/Y3_WORK/y3-wl_image_sims/bin/run-sim

cmd="$python $bin $config_file $outdir image.nproc=$cores_per_job image.nobjects=NOBJECTS \
     --record_file $outdir/job_record.pkl --step_names STEPS"
echo $cmd
$cmd
"""



