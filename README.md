# Advantage-Weighted Regression (AWR)

Code accompanying the paper:
"Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning".
The framework provides an implementation of AWR and supports running experiments on standard OpenAI Gym environments.

Project page: https://xbpeng.github.io/projects/AWR/index.html

## Getting Started

Install requirements:

`pip install -r requirements.txt`

and it should be good to go.

## Training Models

To train a policy, run the following command:

``python run.py --env HalfCheetah-v2 --max_iter 20000 --visualize``

- `HalfCheetah-v2` can be replaced with other environments.
- `--max_iter` specifies the maximum number of training iterations.
- `--visualize` enables visualization, and rendering can be disabled by removing the flag.
- The log and model will be saved to the `output/` directory by default. But the output directory can also be specified with `--output_dir [output-directory]`.

## Loading Models

To load a trained model, run the following command:

``python run.py --test --env HalfCheetah-v2 --model_file data/policies/halfcheetah_awr.ckpt --visualize``

- `--model_file` specifies the `.ckpt` file that contains the trained model. Pretrained models are available in `data/policies/`.

## Code

- `learning/rl_agent.py` is the base agent class, and implements basic RL functionalties.
- `learning/awr_agent.py` implements the AWR algorithm. The `_update()` method performs one update iteration.
- `awr_configs.py` can be used to specify hyperparameters for the different environments. If no configurations are specified for a particular environment, than the algorithm will use the default hyperparameter settings in `learning/awr_agent.py`.

## Data
- `data/policies/` contains pretrained models for the different environments.
- `data/logs/` contains training logs for the different environments, which can be used to plot learning curves.
