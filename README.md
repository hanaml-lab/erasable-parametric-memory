# SLATE: exactly erasable parametric memory for personal agents

Reference implementation for *Learning in deployment, forgetting on request: exactly erasable parametric memory for
personal agents* (Manav Agarwal, hanaML, 2026; Nature Machine Intelligence format preprint, at
[hanaml.com/research/papers/erasable-parametric-memory](https://hanaml.com/research/papers/erasable-parametric-memory)).

A personal agent that consolidates what it learns into model weights reads its memory cheaply and offline, but a
person may ask it to forget, and approximate unlearning leaves traces a membership test can find. SLATE keeps the
agent's ledger of facts as the source of truth and compiles it into small adapters, each trained only on a known
shard of the ledger. A deletion request maps through provenance to the shards that saw any part of it; those shards
are retrained from a checkpoint on what remains, so the erased memory is the memory of a ledger that never held the
facts, bit for bit. The hash of the new state is a receipt that anyone holding the remaining ledger can check by
replaying training.

The article's numbers come from a seeded simulation study and are illustrative. Everything in the
[toy run](#toy-run-real-output) below is real output of this code.

## What is here

- **Ledger** (`slate.ledger`): an append-only set of facts with provenance; `closure()` adds every fact derived from a
  request's episodes, and `value_as_of()` reads what the ledger held at the end of a week.
- **Planner** (`slate.planner`): a fact's tier (weights, or text for sensitive facts) and its shard (a keyed hash of
  the person it is about) are functions of that fact alone, so removing a record never moves another.
- **Rendering** (`slate.render`): each stage's training text is built from the shard's own facts; a fact's
  renderings read only its own slot, and a profile (a derived artefact) is re-derived, never cached.
- **Staged training** (`slate.train`, `slate.model`): LoRA adapters on a small frozen causal LM, trained stage by stage
  with a checkpoint (adapter, optimiser moments) after each; data order comes from generators keyed by
  (shard, stage, epoch), the learning-rate schedule is local to a stage, and kernels are deterministic.
- **Memory** (`slate.memory`): `SlateMemory.consolidate()`, `answer()` (route to an adapter, fall back to the ledger
  below a confidence threshold), `erase()` (closure, affected shards, restore, retrain, receipt) and `replay()`.
- **Audit and baselines** (`slate.mia`, `slate.unlearn`): a calibrated membership statistic with a bootstrap that
  resamples people, and gradient ascent and NPO on a single adapter for comparison.
- **Toy** (`slate.toy`, `slate.demo`): fictitious contacts told to an assistant over three weekly cycles.

Erasure is exact when four things hold (Theorem 1 of the article): planning is local, rendering is local, training
is staged and deterministic, and nothing else that reads the ledger is trained on the person's data. Here the last
holds by construction, since the router is recomputed from the ledger, and the tests check the other three:
`tests/test_train.py` shows that a learning-rate schedule over a shard's whole run breaks incremental retraining, and
`tests/test_memory.py` that a cache of derived renderings, or erasing only the named record of an episode, breaks
the receipt.

This release does not include the large-model harness (Qwen3-14B readers, the LEDGER read path, MLX and ExecuTorch
runtimes) used in the article; `configs/paper.yaml` records its settings.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or any torch >= 2.2
pip install -e ".[dev]"                                              # Python 3.11+; add ".[paper]" for the figures
```

## Quickstart

```python
from slate import Plan, SlateMemory, TrainConfig
from slate.demo import build

vocab, world, model, _ = build(n_contacts=40, weeks=3, seed=0, pretrain_steps=150)   # toy base model and facts
mem = SlateMemory(model, vocab, Plan(n_shards=4), TrainConfig(), weeks=3)
mem.consolidate(world.ledger)

receipt = mem.erase(rids=sorted(mem.ledger.slot_records("piet petrov")), label="forget piet petrov")
assert receipt.state_hash == SlateMemory.replay(mem, mem.ledger).state_hash()   # the receipt checks out
```

```bash
python -m slate.demo                     # the toy below (about a minute on one CPU core)
python scripts/audit.py --seeds 0 1 2 3  # membership audit over four seeds
python examples/erase_your_own.py examples/data/ledger.jsonl --forget "tomas brandt" --attribute city
pytest -q                                # 25 tests
bash scripts/reproduce_paper.sh          # regenerate every table and figure of the article and check its numbers
```

## Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the article's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that exact erasure works end to end and can be verified, on a problem small enough to replay; it does not
reproduce the article's experiments.

- Command: `python -m slate.demo` (settings in `configs/toy.yaml`)
- Code: commit c9c13a8
- Environment: Python 3.13.9, torch 2.12.0 on the CPU with one thread; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 1 min 4 s

### The toy setting

Forty fictitious contacts are told to an assistant over three weekly cycles: five of each contact's six ordinary
attributes, 200 facts in all, and for some contacts a sensitive one, which the planner keeps as text. The base model
is a two-layer transformer pretrained on 160 public people and then frozen. Each contact's facts go to one of four
shards by a keyed hash of the contact; each shard trains one LoRA adapter, stage by stage, with a checkpoint after
each stage. Five deletion requests follow: four contacts in full and one fact of a fifth contact, whose other facts
stay. A replay trains a fresh memory from scratch on the remaining ledger and compares hashes. The membership audit
erases eight more contacts in one request and compares the twelve erased contacts' facts with the facts of twenty
people never mentioned, using the log-probability of each value standardised against the attribute's possible
values and maximised over the shards' adapters.

### Output

```
toy: 40 contacts; 200 facts consolidated into 4 shards over 3 weekly stages; 13 sensitive facts stay as text
base model: 123,968 parameters, pretrained on 960 public facts (8.0 s)
consolidation: 12.6 s; shard sizes 30, 45, 60, 65 facts; 32,768 adapter parameters per shard
  recall from weights alone 0.995; with the fallback to the ledger 1.000 (0.5% of questions fell back)
5 deletion requests (21 consolidated facts):
  forget bruno toivonen                  5 records  shard 0  stages 1-3   1.56 s   receipt e58ce27e2015
  forget jonah yilmaz                    6 records  shard 2  stages 1-3   2.76 s   receipt 905831cffe23
  forget piet petrov                     5 records  shard 2  stages 1-3   2.33 s   receipt 655d815a6c8f
  forget kalia varga                     5 records  shard 2  stages 1-3   2.38 s   receipt e6074f58ac0e
  forget dalia castell's pet             1 records  shard 1  stages 3-3   1.68 s   receipt fc8603edb40e
replay from scratch on the remaining ledger (12.1 s): state hash equal to the erased memory: True (4/4 shards); the five erasures took 10.7 s
  erased facts answered after erasure: 0/21; retained-fact recall 0.994
  the last request again, with dalia castell's profile cached from before it: shard hash equal to the replay False; the shard's own profile of dalia castell still contains the erased value: True (after the exact erasure: False)
membership audit (5.0 s): 60 erased facts about 12 people against 120 facts about 20 people never mentioned (calibrated loss attack, best adapter)
  SLATE after erasure: AUC 0.47 [0.40, 0.55, people resampled]; the same facts before erasure: AUC 0.93
one adapter for all 200 facts: recall 0.930 (6.9 s); approximate unlearning of the same facts:
  gradient ascent  AUC 0.64 (two-sided); erased facts still recalled 3/60; retained recall 0.050; 0.49 s
  NPO + retain     AUC 0.74 (two-sided); erased facts still recalled 8/60; retained recall 0.629; 0.51 s
total 58.8 s
```

### Across seeds

`python scripts/audit.py --seeds 0 1 2 3` repeats the toy on four freshly drawn worlds (6 min 38 s):

```
seed   hash      SLATE AUC [95% CI]  before     GA    NPO
   0   True       0.47 [0.40, 0.55]    0.93   0.64   0.74
   1   True       0.42 [0.34, 0.50]    0.94   0.54   0.52
   2   True       0.51 [0.41, 0.62]    0.98   0.73   0.90
   3   True       0.53 [0.45, 0.61]    0.98   0.76   0.87
```

### What it shows

- Every erasure's receipt matched a replay from scratch on the remaining ledger, on every seed: the erased memory is
  the memory of a ledger that never held the facts. Each request retrained one shard, from the stage of its earliest
  record: stages 1-3 for a contact told about from the first week, stage 3 alone for a fact recorded in the last.
- No erased fact was answered after erasure, and the facts that stayed were still recalled (0.994).
- Keeping a cached profile (a rendering derived from several facts) across the request broke the receipt, and the
  shard still wrote the erased value when asked for that contact's profile: the closure, not the named record, is
  what a request has to remove.
- After exact erasure the audit's AUC stayed near chance on every seed (0.42-0.53, each interval containing 0.5);
  before erasure the same attack separated the facts at 0.93-0.98. The intervals are wide because the audit rests
  on twelve people.
- Approximate unlearning on one adapter holding all 200 facts left the erased facts more distinguishable on three
  seeds of four (AUC 0.64-0.90), still recalled some of them (3 and 8 of 60 on seed 0) and damaged the facts that
  should have stayed (retained recall 0.050 after gradient ascent, 0.629 after NPO, from 0.930).
- The toy is far smaller than the article's setting: its base model has 0.12M parameters and its shards hold 30-65
  facts, so its times say nothing about a 14B reader on a laptop.

## Results from the article

From the article's seeded simulation (illustrative; Qwen3-14B reader; mean over 5 seeds). Question answering, with
the same consolidated slots placed in weights by each method and the same calibrated fallback to the ledger:

| Method | LongMemEval-S (%) | Read tokens | LoCoMo J |
|---|---|---|---|
| Retrieval only (LEDGER with TIDE's schedule) | 70.2 | 4.70k | 70.6 |
| Full LoRA (one adapter) | 68.9 | 0.98k | 68.6 |
| Sparse memory fine-tuning | 69.3 | 0.92k | 69.2 |
| SISA, random shards | 68.4 | 1.08k | 69.0 |
| **SLATE** | **71.8** | **0.77k** | **71.5** |

Erasure on EraseStream (1,000 deletion requests over 40 synthetic user-years; time on a laptop with an Apple M4 Max):

| Method | Membership-inference AUC | Median time per erasure | Retained-fact accuracy after the year (%) |
|---|---|---|---|
| Retrieval only | 0.50 | 0.4 s | 80.3 |
| Full LoRA + gradient ascent / NPO / SimNPO | 0.68 / 0.63 / 0.65 | 19 s / 49 s / 37 s | 71.7 / 75.0 / 75.8 |
| Sparse memory fine-tuning + slot reset | 0.61 | 25 s | 78.0 |
| Full LoRA, retrained from scratch | 0.51 | 1.1 h | 79.6 |
| SISA, random shards | 0.50 | 1.5 min | 78.6 |
| **SLATE** | **0.50** | **38 s** | **81.3** |

SLATE matches retrieval on LongMemEval-S (paired difference +1.6 points, 95% interval −0.4 to +3.6) with 6.1 times
fewer read tokens, and every exact method's state matched a replay from scratch on all 1,000 requests. These numbers
come from the article's seeded simulation and are illustrative.

## Reproduce the article's tables and figures

`paper/make_data.py` is the seeded generator of every number in the article and `paper/figures.py` draws every figure
from its output. `paper/check_results.py` compares the regenerated tables and results with the digests in
`paper/results.sha256`; CI runs it with numpy 2.3.5 and one OpenBLAS thread, the environment the digests were
recorded in.

```bash
bash scripts/reproduce_paper.sh
```

## Layout

```
src/slate/   ledger.py  planner.py  render.py  model.py  train.py  memory.py  mia.py  unlearn.py  toy.py  demo.py
configs/     toy.yaml  paper.yaml
scripts/     run_toy.py  audit.py  reproduce_paper.sh
examples/    erase_your_own.py  data/ledger.jsonl
paper/       make_data.py  figures.py  check_results.py  results.sha256  toy_run.md
tests/       test_ledger.py  test_planner.py  test_render.py  test_train.py  test_memory.py  test_mia.py  test_demo.py
```

## Citation

```bibtex
@techreport{agarwal2026slate,
  author      = {Agarwal, Manav},
  title       = {Learning in deployment, forgetting on request: exactly erasable parametric memory for personal agents},
  institution = {hanaML},
  year        = {2026},
  url         = {https://hanaml.com/research/papers/erasable-parametric-memory}
}
```

## Licence

Apache-2.0.
