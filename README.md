# Hybrid kernel experiment using LIBLINEAR model

## Requirements
```text
pip install -U liblinear-official numpy scikit-learn scipy
```
## Run
To run the hybrid kernel experiments (train and test) using LIBLINEAR model, you can set a series of args from command line. More details of each argument are located at `Option_for_cmd.py`.
We recommend that you set the following parameters:
`--sketch_method`: the kernel method to generate sketch, for exmple, `SM, MM, RBF, Poly`
`--portion`: the feature partition ratio
`--dataset_file_name`: dataset
`--k`: the total sketch length
`--C`: the regularization parameter for LIBLINEAR model
`--c`: the compression factor

```bash
python Experiments.py --sketch_method SM --portion 0.3 --dataset_file_name kits --k 1024 --C 1 --c 16
```

And results(such as accuracy and running time) will be stored in corresponded directory.
The above runtime results are from a single experiment with fixed parameters. All the experimental results in our paper are the averages obtained from a large number of repeated experiments and fine-tuning of parameters. If you wish, you can adjust the parameter values to achieve better results.
