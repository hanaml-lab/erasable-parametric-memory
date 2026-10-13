# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the article's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that exact erasure works end to end and can be verified, on a problem small enough to replay; it does not
reproduce the article's experiments.

- Command: `python -m slate.demo` (settings in `configs/toy.yaml`)
- Code: commit c9c13a8
- Environment: Python 3.13.9, torch 2.12.0 on the CPU with one thread; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 1 min 4 s

## The toy setting

Forty fictitious contacts are told to an assistant over three weekly cycles: five of each contact's six ordinary
attributes, 200 facts in all, and for some contacts a sensitive one, which the planner keeps as text. The base model
is a two-layer transformer pretrained on 160 public people and then frozen. Each contact's facts go to one of four
shards by a keyed hash of the contact; each shard trains one LoRA adapter, stage by stage, with a checkpoint after
each stage. Five deletion requests follow: four contacts in full and one fact of a fifth contact, whose other facts
stay. A replay trains a fresh memory from scratch on the remaining ledger and compares hashes. The membership audit
erases eight more contacts in one request and compares the twelve erased contacts' facts with the facts of twenty
people never mentioned, using the log-probability of each value standardised against the attribute's possible
values and maximised over the shards' adapters.

## Output

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

## Across seeds

`python scripts/audit.py --seeds 0 1 2 3` repeats the toy on four freshly drawn worlds (6 min 38 s):

```
seed   hash      SLATE AUC [95% CI]  before     GA    NPO
   0   True       0.47 [0.40, 0.55]    0.93   0.64   0.74
   1   True       0.42 [0.34, 0.50]    0.94   0.54   0.52
   2   True       0.51 [0.41, 0.62]    0.98   0.73   0.90
   3   True       0.53 [0.45, 0.61]    0.98   0.76   0.87
```

## What it shows

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
