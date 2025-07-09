# # graph regression
# python ./experiments/train_zinc.py --abs-pe rw --se gnn --gnn-type pna2 --dropout 0.3 --k-hop 3 --use-edge-attr --model graphvit
# python ./experiments/train_zinc.py --abs-pe rw --se gnn --gnn-type pna2 --dropout 0.3 --k-hop 3

# python ./experiments/train_zinc.py --abs-pe rw --se khopgnn --gnn-type pna2 --dropout 0.2 --k-hop 3 --use-edge-attr

# # node classification
# python ./experiments/train_SBMs.py --dataset PATTERN --weight-class --abs-pe rw --abs-pe-dim 7 --se gnn --gnn-type pna3 --dropout 0.2 --k-hop 3 --num-layers 6 --lr 0.0003

# python ./experiments/train_SBMs.py --dataset CLUSTER --weight-class --abs-pe rw --abs-pe-dim 3 --se gnn --gnn-type pna2 --dropout 0.4 --k-hop 3 --num-layers 16 --dim-hidden 48 --lr 0.0005

# graph classification
# Train SAT on OGBG-PPA
# python experiments/train_ppa.py --dataset ogbg-ppa --epoch 200 --lr 0.003 --gnn-type gcn --use-edge-attr --use_gcn --abs-pe rw
# python experiments/train_ppa.py --dataset ogbg-ppa --num-layers 3 --dim-hidden 128 --dropout 0.1 --epochs 200 --lr 0.0003 --weight-decay 0.0001 --batch-size 32 --warmup 10 --edge-dim 128 --gnn-type graph --use-edge-attr --model graphvit --abs-pe rw --plot_attn
# python experiments/train_ppa.py --dataset ogbg-ppa --num-layers 3 --dim-hidden 128 --dropout 0.1 --epochs 200 --lr 0.0003 --weight-decay 0.0001 --batch-size 32 --warmup 10 --edge-dim 128 --gnn-type graph --use-edge-attr --model graphvit --abs-pe rw
# python experiments/train_ppa.py --gnn-type gcn --use-edge-attr --model graphvit

# Train SAT on OGBG-CODE2
# python ./experiments/train_code2.py --gnn-type gcn --use-edge-attr
# python ./experiments/train_code2.py --gnn-type gcn --use-edge-attr --model graphvit


# python ./experiments/train_graph_classification.py --gnn-type gcn --model graphvit --DS MUTAG --abs-pe rw --epoch 10 --use_gcn --plot_attn --unsupervised
# python ./experiments/train_graph_classification.py --gnn-type gcn --model graphvit --DS MUTAG --abs-pe rw --epoch 10 --use_gcn --plot_attn
python ./experiments/train_graph_classification.py --gnn-type gcn --model graphvit --DS MUTAG --abs-pe rw --epoch 10 --use_gcn
# python ./experiments/train_graph_classification.py --gnn-type gcn --model graphvit --DS MUTAG --abs-pe rw --epoch 200 --use_pretrained_gin