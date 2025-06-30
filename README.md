# BI
usefull commands 
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTHONPATH=$(pwd)
conda activate bd
tmux new -s step_train
python codes/step.py | tee logs/steps.log
ssh jag@172.17.38.139
python codes/Oraclefinetune.py | tee logs/Oracle_train.log