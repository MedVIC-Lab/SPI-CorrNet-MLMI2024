# SPI-CorrNet MLMI MICCAI 2024
[Probabilistic 3D Correspondence Prediction from Sparse Unsegmented Images](https://arxiv.org/abs/2407.01931)
To run training and inference 
```
python run_SPI_CorrNet.py --config configs/liver.json
```

To run with point cloud data, replace `idx` to `None` in the `run_SPI_CorrNet.py`


Data organization: 
```
train
	- meshes 
	- images 
val
	- meshes
	- images 
test 
	- meshes
	- images 
mean.particles  
```

TODO: Upload dataset to SW Cloud portal and provide links
