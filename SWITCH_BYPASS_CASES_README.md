# SwitchOptimizer Bypass Cases

This note summarizes the four bypass cases currently used in `SwitchOptimizer`
Phase 2, with emphasis on the special behavior of `Switch` chains. The examples
are intentionally minimal: only the local `Switch` inputs that reach the
consumer are shown. Upstream `Merge`, `Identity`, and unrelated graph
structure are omitted.

## Notation

- `S(p)[1]` means the true output of `Switch(data, p)`.
- `S(p)[0]` means the false output of `Switch(data, p)`.
- `D(t)` means the `DeadnessAnalysis` predicate printed in logs for tensor `t`.
  In practice this is the condition under which the tensor is live.
- `Guard(t)` means the logical branch condition that keeps tensor `t` live.
  For a chain, this is usually a conjunction of branch literals.
- `Consumer(...)` means an ordinary compute node such as `Add` or `Mul`, not a
  control-flow node.

For a single true-branch `Switch`, the local deadness is:

$$
D(S(p)[1]) = D(data) \land p
$$

When the data input is otherwise live in the local picture, this reduces to:

$$
D(S(p)[1]) = p
$$

## Why Switch Chains Are Special

For a chain, the effective guard is not a single predicate but the conjunction
of all branch choices along the chain.

### Example: true-only chain

```text
x -> S(p0)[1] -> S(p1)[1] -> S(p2)[1]
```

The local guard is:

$$
Guard(t) = p0 \land p1 \land p2
$$

### Example: mixed true and false outputs

```text
x -> S(p0)[1] -> S(p1)[0] -> S(p2)[1]
```

The local guard is:

$$
Guard(t) = p0 \land \lnot p1 \land p2
$$

So the special point is not "how many Switches are there" by itself. The real
semantic object is the set of branch literals carried by the chain.

In Phase 1, repeated chains may be folded into one surviving `Switch` whose
predicate input is a Boolean DAG made from `LogicalAnd` and `LogicalNot`.
Conceptually, the folded predicate still represents the same literal set. For
example:

```text
x -> S(p0)[1] -> S(p1)[1] -> S(p2)[1]
```

can be treated as a folded guard:

$$
Guard(t) = p0 \land p1 \land p2
$$

The key idea behind all four cases is therefore:

- Case 1 and Case 2 compare equal guards.
- Case 3 and Case 4 compare stronger vs. weaker guards.
- For folded conjunction-only chains, "stronger" means a strict superset of
  literals.

This is also why folded unequal-depth cases are special: raw `output_deadness`
may contain fresh internal Switch atoms, while the structural folded guard is
still a clean conjunction such as `{p0, p1, p2, p3}` vs. `{p0, p1, p2}`.

### Current folded examples: why raw deadness is not enough

The current repro models `SWITCH_INDIRECT_FOLD_STRICT_STRONGER_INDIRECT_4V3`
and `SWITCH_INDIRECT_FOLD_STRICT_STRONGER_DIRECT_3V4` show the real issue.

Before Phase 1 folding, the chain meaning is easy to read:

- 4v3 means the indirect witness side carries a guard like
  $p0 \land p1 \land p2 \land p3$ while the direct side carries
  $p0 \land p1 \land p2$.
- 3v4 means the direct side carries $p0 \land p1 \land p2 \land p3$ while the
  witness side carries $p0 \land p1 \land p2$.

So semantically, the containment relation is obvious from the chain guards.

After Phase 1 rewriting, however, the surviving `Switch` no longer sees the
original chain directly. Its predicate input becomes a newly created folded
tensor such as:

```text
swo_swo_pred_id0_and_pred_id1_and_pred_id2
```

or

```text
swo_swo_swo_pred_id0_and_pred_id1_and_pred_id2_and_pred_id3
```

Then `DeadnessAnalysis` computes `output_deadness` for the `Switch` output in
terms of these new folded predicate tensors. In the logs this appears as fresh
atoms such as `*swo_swo...:0` and `*swo_swo_swo...:0`.

The important consequence is:

- the original relation
  $p0 \land p1 \land p2 \land p3 \Rightarrow p0 \land p1 \land p2$
  is still true semantically,
- but the raw `output_deadness` strings no longer expose that relation in a
  directly comparable form,
- so `CanProveImplication` over the deadness predicates may return false in
  both directions even though the folded chain guards clearly have a strict
  subset or superset relation.

That is why folded chain cases need an additional path besides raw deadness
implication:

1. trace the witness back to the upstream `Switch`,
2. read the surviving folded predicate input,
3. extract the literal set from the folded Boolean DAG,
4. compare the literal sets directly.

So the special handling for `Switch` chains is not just "chains are deeper".
It is specifically that the Phase 1 rewrite introduces new folded predicate
tensors, and those new tensors can hide the original implication relation from
the raw `output_deadness` view.

## Case 1: Direct-Direct Equality

### Minimal local graph

```text
x0 -> S(p0)[1] -> S(p1)[1] --\
                                 +--> Consumer
x1 -> S(p0)[1] -> S(p1)[1] --/
```

### Deadness view

$$
D(d_0) = p0 \land p1
$$

$$
D(d_1) = p0 \land p1
$$

Both direct inputs carry the same chain guard. Keeping one direct `Switch`
input is enough to keep the consumer under the same deadness condition.

### Rewrite

- Keep one direct `Switch` output as the anchor.
- Replace the other direct input with its raw data input.

```text
x0 -> S(p0)[1] -> S(p1)[1] --\
                                 +--> Consumer
x1 -----------------------------/
```

## Case 2: Direct-Other Equality

### Minimal local graph

```text
x0 -> S(p0)[1] -> S(p1)[1] --\
                                 +--> Consumer
w ------------------------------/
```

Only the local shape is shown. The upstream path that produces `w` is omitted,
but `w` is assumed to already be under the same folded chain guard.

### Deadness view

$$
D(d) = p0 \land p1
$$

$$
D(w) = p0 \land p1
$$

The non-`Switch` sibling already carries the same deadness as the direct
`Switch` chain output, so the direct `Switch` is redundant as a deadness
carrier.

### Rewrite

- Keep `w` as the witness.
- Replace the direct chain output with its raw data input.

```text
x0 -----------------------------\
                                 +--> Consumer
w ------------------------------/
```

## Case 3: Direct-Direct Strict Implication

### Minimal local graph

```text
x0 -> S(p0)[1] -> S(p1)[1] -> S(p2)[1] --\
                                             +--> Consumer
x1 -> S(p0)[1] -> S(p1)[1] ----------------/
```

This example shows the stronger direct input on the left and the broader direct
input on the right.

### Deadness view

$$
D(d_0) = p0 \land p1 \land p2
$$

$$
D(d_1) = p0 \land p1
$$

and therefore:

$$
p0 \land p1 \land p2 \Rightarrow p0 \land p1
$$

So the left chain is stricter. If the consumer is live through `d0`, then `d1`
must already be live. The broader direct chain is redundant.

### Rewrite

- Keep the stricter direct input.
- Bypass the broader direct input.

```text
x0 -> S(p0)[1] -> S(p1)[1] -> S(p2)[1] --\
                                             +--> Consumer
x1 ----------------------------------------/
```

## Case 4: Direct-Indirect Strict Implication

### Minimal local graph

```text
x0 -> S(p0)[1] -> S(p1)[1] -> S(p2)[1] --\
                                             +--> Consumer
w ------------------------------------------/
```

Again, only the local consumer view is shown. The upstream path that produces
`w` is omitted. The important point is that `w` comes from a stronger chain on
another path.

### Deadness view

$$
D(d) = p0 \land p1 \land p2
$$

$$
D(w) = p0 \land p1 \land p2 \land p3
$$

and therefore:

$$
p0 \land p1 \land p2 \land p3 \Rightarrow p0 \land p1 \land p2
$$

The indirect sibling is stricter than the direct chain output. That means the
consumer is already gated by the stronger witness `w`, so the direct `Switch`
chain is redundant.

### Rewrite

- Keep `w` as the witness.
- Replace the direct chain output with its raw data input.

```text
x0 ----------------------------------------\
                                             +--> Consumer
w ------------------------------------------/
```

## Equality vs. Strict Implication

- Case 1 and Case 2 are equality cases: one surviving input already has the
  same deadness as the bypass candidate.
- Case 3 and Case 4 are strict implication cases: one surviving input has a
  stronger deadness predicate than the bypass candidate.
- In folded conjunction-only patterns, a stronger predicate is recognized by a
  strict superset of literals, for example `{p0, p1, p2, p3}` is stronger than
  `{p0, p1, p2}`.
- So for chains, the important rule is not "4 hops beats 3 hops" by itself.
  The real rule is: the 4-hop side is stronger only because its guard carries
  one extra literal.

## What Is Not Shown Here

- `Merge`, `Identity`, and export-only nodes are intentionally omitted.
- For Case 2 and Case 4, the witness input may come through a longer upstream
  path; only its final local role as a non-`Switch` sibling input matters in
  this README.
- These examples assume the consumer is a normal compute op. Control-flow nodes
  such as `Switch` and `Merge` are excluded from this bypass phase.
- The real logs may still print `output_deadness` with extra internal `*swo...`
  atoms. For folded unequal chains, the optimizer may rely on structural guard
  comparison rather than raw deadness implication to prove containment.