#!/bin/bash

#SBATCH --job-name=AMP
#SBATCH --account=nn2977k
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=slurm-test.out

## Set up job environment
source /cluster/bin/jobsetup

# Run job
module load Anaconda3/5.1.0
source activate md
echo $1
cd $1
python test_calc.py

exit 1
