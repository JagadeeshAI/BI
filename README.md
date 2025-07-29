# BI
usefull commands 
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTHONPATH=$(pwd)
conda activate bd
python codes/step.py | tee logs/steps.log
python codes/Oraclefinetune.py | tee logs/OF.log
ssh jag@172.17.38.139
tmux attach -t step_train
tmux new -s step_train
python baseline/students.py | tee logs/student.log
python steps.py | tee logs/clpu.log
python steps.py | tee logs/er_ewc.log
python unlearn.py | tee l2ul.log