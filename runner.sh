#!/bin/bash
# -----------------------------------------
# Run all models for problem_pooling
# -----------------------------------------
# This script will sequentially run:
#   1. MLP
#   2. PINN
#   3. KKT-HardNet
# for the directory ./Op_problems/problem_pooling
# and plot the losses after each run.
# -----------------------------------------

# PROBLEM_DIR="./Op_problems/problem_pooling" # Put your problem directory
PROBLEM_DIR="./Op_problems/problem_tp8_113"
DO_PLOT=0

echo "📁 Problem Directory: $PROBLEM_DIR"

# If you do not want to run a specific model just comment it out
python main.py --dir_path "$PROBLEM_DIR" --mode mlp  --do_plot $DO_PLOT
python main.py --dir_path "$PROBLEM_DIR" --mode pinn --do_plot $DO_PLOT

DO_PLOT=1

python main.py --dir_path "$PROBLEM_DIR" --mode kkt  --do_plot $DO_PLOT


echo "✅ Code have been executed for $PROBLEM_DIR"
