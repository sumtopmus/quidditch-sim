# Campaign Goal: <name>

**Objective:** <metric> <comparator> <target>
(e.g. `eval/success_rate` (Blue vs league) > 0.50, judged on honest success
rate de-noised over ≥100 episodes)

**Mode:** interactive | autonomous
**autonomous_allow_code:** true | false   # false => config-only when autonomous

**Budget:**
- max_iterations: <N>
- max_wallclock: <e.g. 8h>
- per_run_step_cap: <e.g. 10_000_000>

**Levers in scope:**
- <reward terms / weights>
- <curriculum knobs>
- <obs blocks>
- <opponent mixture>
- code changes in scope: yes | no

**Baselines** (for collapse-vs-baseline rules):
- <label>: <entity/project/run_id>

**Out of scope / do not touch:**
- <constraints>
