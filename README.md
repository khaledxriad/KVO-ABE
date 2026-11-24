# KVO-ABE: prototype and measurement data

Reference prototype and raw measurement data for the article
**"KVO-ABE: Keyless Verifiable Outsourced Decryption for Lattice Attribute-Based Encryption"**
by Khaled Riad (King Faisal University).

KVO-ABE lets an untrusted cloud do the policy-dependent part of lattice attribute-based decryption
using public data only. The data user finishes with a short pass of inner products and a hash, and
rejects any wrong answer from the cloud by checking a 32-byte tag.

The code is a research prototype. It runs at toy lattice dimensions (n = 16, 32, 64) with a 62-bit
modulus, and key generation uses a shortcut that is described in the article. It is not meant for
protecting real data.

## Contents

| Path | What it is |
| --- | --- |
| `kvo.py` | The scheme: setup, key-homomorphic public evaluation, encryption, cloud evaluation, user finalization, tag and pad. |
| `exp.py` | The experiments E1 (correctness), E2 (timing, five repetitions), E3 (cheating cloud and tolerance sweep). |
| `model.py`, `params.py` | A simplified primal-attack estimator and the three 128-bit parameter sets P-I, P-II, P-III. |
| `sizes.py` | The size model for the comparison with pairing-based schemes. |
| `tests/test_smoke.py` | A quick check that honest decryption works and that a tampered answer is never accepted as a wrong message. |
| `data/` | CSV files and logs produced by the scripts; these are the numbers reported in the article. |

## Requirements

Python 3.10 or later with NumPy (see `requirements.txt`). The reported measurements were taken with
Python 3.13 and NumPy 2.5 on a two-core virtual machine.

```
pip install -r requirements.txt
```

## Reproducing the results

```
python3 -m pytest -q tests          # seconds
python3 params.py                   # writes data/params_sets.csv
python3 sizes.py                    # writes data/e5_sizes.csv
python3 exp.py e1                   # correctness: 210 honest decryptions
python3 exp.py e3                   # 60,000 cheating trials and the tolerance sweep
python3 exp.py e2                   # timing; runs the whole experiment five times, takes a while
```

E1, E3, the parameter sets, and the size model are deterministic (fixed seeds) and reproduce the CSV
files in `data/` exactly. Timings in E2 depend on the machine. `data/e2_timing_runs.csv` holds the
individual repetitions and `data/e2_timing.csv` the medians, with the smallest and largest speed-up
over the five repetitions.

## Data files

| File | Content |
| --- | --- |
| `e1_correctness.csv` | Trials, successes, largest observed noise and worst-case bound, per configuration. |
| `e2_timing.csv`, `e2_timing_runs.csv` | Cloud, user (outsourced), and user (local) times; sizes of the auxiliary data and ciphertexts. |
| `e3_cheating.csv` | Outcomes (accepted correct, accepted wrong, rejected) for each cloud behavior. |
| `e3_tolerance.csv` | Sweep of perturbation magnitude against the acceptance outcome. |
| `params_sets.csv` | Parameter sets for the 128-bit estimate. |
| `e5_sizes.csv` | Pairing-based size model and lattice sizes for 4 to 64 attributes. |
| `*.log` | Console output of the runs. |

## Citation

See `CITATION.cff`. The archived release is on Zenodo: [I will set it later].

## License

MIT, see `LICENSE`.
