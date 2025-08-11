#!/bin/bash
#SBATCH --job-name=capybarabenchmarkpacha
#SBATCH --output=/scratch/s/shirimb/msep/capybara/capybarabenchmark.txt
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --account=def-shirimb

# Load necessary modules
module load NiaEnv/2022a
module load gcc/11.3.0
module load openblas
module load r/4.3.3-batteries-included

export CAPYBARA_PORTABLE="no"
export CAPYBARA_USE_FAST_MATH="no"

export LAPACK_LIBS="${LD_PRELOAD}"
export BLAS_LIBS="${LD_PRELOAD}"
export R_LIBS_USER=/home/s/shirimb/msep/scratch/capybara/rpkgs
export PATH=/home/s/shirimb/msep/scratch/capybara/R/bin/:$PATH

module show openblas
export OPENBLAS_LIB="/gpfs/fs1/scinet/niagara/software/2022a/opt/gcc-11.3.0/openblas/0.3.15/lib/libopenblas_skylakex-r0.3.15.so"
export LD_PRELOAD="${OPENBLAS_LIB}"

# run this on local side
# rsync -av --update /home/pacha/github/capybara_a.b.c .tar.gz msep@niagara.scinet.utoronto.ca:~/scratch/capybara/capybara_a.b.c.tar.gz

# run this outside the sbatch environment once
# Rscript -e "install.packages(c('kendallknight','cpp11armadillo','Formula','data.table','tradepolicy','alpaca','bench','janitor'), repos='https://cloud.r-project.org/')"
# export CAPYBARA_PORTABLE="no"
# export CAPYBARA_USE_FAST_MATH="yes"
# R CMD INSTALL cpp11armadillo_a.b.c.tar.gz
# R CMD INSTALL capybara_a.b.c.tar.gz

Rscript benchmarks.r
