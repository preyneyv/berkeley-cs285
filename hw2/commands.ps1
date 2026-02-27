uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na



# No baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
uv run src/scripts/run.py --env_name HalfCheetah-v4 -n 100 -b 5000 -eb 3000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline_lower_blr_bgs

# GAE Lambda {0, 0.95, 0.98, 0.99, 1}
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 `
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline `
  --gae_lambda 0 --exp_name lunar_lander_lambda0

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 `
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline `
  --gae_lambda 0.95 --exp_name lunar_lander_lambda0.95

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 `
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline `
  --gae_lambda 0.98 --exp_name lunar_lander_lambda0.98

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 `
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline `
  --gae_lambda 0.99 --exp_name lunar_lander_lambda0.99

uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 `
  -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline `
  --gae_lambda 1 --exp_name lunar_lander_lambda1


# Hyperparameter Tuning
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum
# uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 150 -b 1000 -eb 1000 --exp_name pendulum_tune1 `
  -rtg -na --discount 0.99 -lr 0.002 


uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 150 -b 1000 -eb 1000 `
  -rtg -na --discount 0.99 -lr 0.003 --exp_name pendulum_tune2

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 150 -b 1000 -eb 1000 `
  -rtg -na --use_baseline --gae_lambda 0.95 -lr 0.002 -blr 0.005 -bgs 3 `
  --discount 0.99 --exp_name pendulum_tune3

uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 180 -b 800 -eb 1000 `
  -rtg -na --discount 0.99 -lr 0.003 --exp_name pendulum_tune4
