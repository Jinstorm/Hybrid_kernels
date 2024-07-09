# Secure vertical logistic regression
This is the implementation of [CAESAR](https://arxiv.org/abs/2008.08753) and [SecureML](https://eprint.iacr.org/2017/396.pdf).
## Requirements
```text
pip install numpy scikit-learn gmpy2 scipy
```
## Run
To run **CAESAR**, you can set a series of args from command line. More details of each argument are located at function `parse_input_parameter()` in `logistic_regression/mini_batch_logistic/CAESAR.py`.
```bash
python3 -u CAESAR.py -d kits -p 37 -m sketch -a pminhash -k 1024 -c 2 -b 2 -o bin -r mm -l off -s linear -al 0.005 -lm 1 -cp 1 -i 150 -t 128 -e 150 -f
```
To run **SecureML**, following the command below:
```bash
python3 -u secureML.py -d kits -p 37 -m raw -a pminhash -k 1024 -c 4 -b 2 -o bin -r mm -l off -s linear -al 0.0001 -lm 1 -i 100 -t 8 -e 100 -f
```
And results(such as accuracy and running time) will be stored in corresponding directory.

In our implementation, we adopt some basic modules from [FATE](https://github.com/FederatedAI/FATE.git), which is the world's first industrial grade federated learning open source framework, and attach the license to related files or functions.