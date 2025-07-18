# BI
usefull commands 
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTHONPATH=$(pwd)
conda activate bd
python codes/step.py | tee logs/steps.log
python codes/Oraclefinetune.py | tee logs/OF.log
ssh jag@172.17.38.139
tmux attach -t step_train