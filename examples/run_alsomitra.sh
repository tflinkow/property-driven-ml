#!/bin/bash
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1"
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=DL2
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=GD
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=LK
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=RC
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=YG
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=STL
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty1" --experiment-name="AlsomitraProperty1" --logic=LL

python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty2" --experiment-name="AlsomitraProperty2"
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty2" --experiment-name="AlsomitraProperty2" --logic=DL2
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty2" --experiment-name="AlsomitraProperty2" --logic=GD
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty2" --experiment-name="AlsomitraProperty2" --logic=STL
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty2" --experiment-name="AlsomitraProperty2" --logic=LL

python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty3" --experiment-name="AlsomitraProperty3"
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty3" --experiment-name="AlsomitraProperty3" --logic=DL2
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty3" --experiment-name="AlsomitraProperty3" --logic=GD
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty3" --experiment-name="AlsomitraProperty3" --logic=STL
python ../main.py --dataset=alsomitra --epochs=100 --batch-size=64 --lr=1e-3 --oracle-steps=60 --oracle-restarts=80 --constraint="AlsomitraProperty3" --experiment-name="AlsomitraProperty3" --logic=LL