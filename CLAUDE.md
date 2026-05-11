# Project notes

Drone Quidditch Sim — a reinforcement-learning project that trains a quadcopter
to fly through a goal hoop on a MuJoCo physics stack, using SB3 PPO. See
`README.md` for the full project description, game rules, observation/action
spaces, reward function, and architecture overview.

## Workflow

Day-to-day actions (train / eval / resume / promote / lineage / tensorboard / repro / demos) live in the TUI launcher: `make ui` (or `python -m tui`). The Makefile keeps only infrastructure targets (install / configs / clean / test*).
