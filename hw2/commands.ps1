uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole \
  -n 100 -b 1000

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_rtg \
  -n 100 -b 1000 -rtg

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_na \
  -n 100 -b 1000 -na

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_rtg_na \
  -n 100 -b 1000 -rtg -na

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_lb \
  -n 100 -b 4000

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_lb_rtg \
  -n 100 -b 4000 -rtg

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_lb_na \
  -n 100 -b 4000 -na

uv run src/scripts/run.py --env_name CartPole-v0 --exp_name cartpole_lb_rtg_na \
  -n 100 -b 4000 -rtg -na




# No baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 --exp_name cheetah \
  -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 

# Baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 --exp_name cheetah_baseline \
  -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5

# Fewer baseline gradient steps
uv run src/scripts/run.py --env_name HalfCheetah-v4 --exp_name cheetah_baseline_lower_bgs \
  -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 1

# GAE Lambda {0, 0.95, 0.98, 0.99, 1}
uv run src/scripts/run.py --env_name LunarLander-v2 --exp_name lunar_lander_lambda0 \
  --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 \
  -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0

uv run src/scripts/run.py --env_name LunarLander-v2 --exp_name lunar_lander_lambda0.95 \
  --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 \
  -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95

uv run src/scripts/run.py --env_name LunarLander-v2 --exp_name lunar_lander_lambda0.98 \
  --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 \
  -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98

uv run src/scripts/run.py --env_name LunarLander-v2 --exp_name lunar_lander_lambda0.99 \
  --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 \
  -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99

uv run src/scripts/run.py --env_name LunarLander-v2 --exp_name lunar_lander_lambda1 \
  --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 \
  -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1


# Hyperparameter Tuning
uv run src/scripts/run.py --env_name InvertedPendulum-v4 --exp_name pendulum \
  -n 100 -b 5000 -eb 1000

uv run src/scripts/run.py --env_name InvertedPendulum-v4 --exp_name pendulum_tune1 \
  -n 150 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.002 

uv run src/scripts/run.py --env_name InvertedPendulum-v4 --exp_name pendulum_tune2 \
  -n 150 -b 1000 -eb 1000 -rtg -na --discount 0.99 -lr 0.003

uv run src/scripts/run.py --env_name InvertedPendulum-v4 --exp_name pendulum_tune3 \
  -n 150 -b 1000 -eb 1000 -rtg -na --use_baseline --gae_lambda 0.95 -lr 0.002 \
  -blr 0.005 -bgs 3 --discount 0.99

uv run src/scripts/run.py --env_name InvertedPendulum-v4 --exp_name pendulum_tune4 \
  -n 180 -b 800 -eb 1000 -rtg -na --discount 0.99 -lr 0.003
