#!/bin/sh

#SBATCH --job-name=qwen_doc_filter
#SBATCH --partition=a100
#SBATCH --output=output
#SBATCH --error=output.err
#SBATCH --nodes=1
#SBATCH --nodelist=ngpu08
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# conda
. "/userspace/bak2/miniconda3/etc/profile.d/conda.sh"
conda activate pytorch_cuda118_env

# cuda
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}

# while job is running, you can exec:
# srun --pty --overlap --jobid <jobid> bash
#
# this will give you cli for that node, where you can use nvidia-smi


# if you have to use pip, you must set cache directory before "pip install":
export PIP_CACHE_DIR="/userspace/bak2/pip/.cache"

# other cache dirs can be set in the same manner:
export HF_HOME="/userspace/bak2/hf"
export HF_TOKEN="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"
#export TRANSFORMERS_CACHE="/userspace/<myUser>/hf_cache"
#export HF_DATASETS_CACHE="/userspace/<myUser>/hf_cache"
python MenoProject/qwen_doc_filter.py --model_path=/storage0/bi/models/Qwen2.5-14B-Instruct/ --dir_path=MenoProject/data --output=MenoProject/filtered_dir --sys_prompt=MenoProject/sys_promptv1.txt
# if you need git to clone a repo, use ssh method (not https!)
