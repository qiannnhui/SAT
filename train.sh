# # graph regression
python ./experiments/train_zinc.py --abs-pe rw --se gnn --gnn-type pna2 --dropout 0.3 --k-hop 3 --use-edge-attr --model groupvit
# python ./experiments/train_zinc.py --abs-pe rw --se gnn --gnn-type pna2 --dropout 0.3 --k-hop 3

# python ./experiments/train_zinc.py --abs-pe rw --se khopgnn --gnn-type pna2 --dropout 0.2 --k-hop 3 --use-edge-attr

# # node classification
# python ./experiments/train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.2 --k-hop 3 --num-layers 6 --lr 0.0003

# python ./experiments/train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0005

# graph classification
# Train SAT on OGBG-PPA
# python experiments/train_ppa.py --gnn-type gcn --use-edge-attr

# Train SAT on OGBG-CODE2
# python ./experiments/train_code2.py --gnn-type gcn --use-edge-attr