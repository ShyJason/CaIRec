# I3clear in MMRec

This directory vendors the standalone I3clear/I3-MRec code so the original I3
baseline can be run from the MMRec repository without depending on an external
checkout.

Run it from the repository root through:

```bash
./run_i3clear.sh --dataset baby --max_info_coeff 1e-3 --min_info_coeff 1e-5 --reg_coeff 1e-3 --penalty_coeff 300 --lr 1e-3 --missing_rate 0.3 --exp_mode mm
```

The vendored code keeps its original relative path behavior:

- data is read from `Data/<dataset>` under the MMRec repository root
- outputs are written to `exp_report/<dataset>/<suffix>` under the MMRec repository root
- the original I3 IRM and IB objectives are enabled by default

For ablations provided by this I3clear source tree:

```bash
./run_i3clear.sh ... --disable_irm 1
./run_i3clear.sh ... --disable_ib 1
```
