#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=00:02:00
#SBATCH --job-name=ED_run_1
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=benjamin.michen@tu-dresden.de
#SBATCH --account=p_qvt

module load release/24.10
module load Anaconda3/2024.02-1

conda init
conda activate /data/horse/ws/s3593425-conda_virtual_environment/conda-env

python -u Transport_kwant.py 
