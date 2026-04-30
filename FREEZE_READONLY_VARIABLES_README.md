# Freeze Readonly Variables Design

This document describes the current `FreezeReadonlyVariablesPass` quick-hack design used to turn readonly TensorFlow variables into graph constants before XLA auto-clustering.

The goal is to make the graph that reaches `MarkForCompilationPass` contain normal tensor constants instead of `VariableV2`, `VarHandleOp`, and selected resource read/gather operations. This helps test whether XLA clustering and codegen behave better when model weights are visible as constants.

## Scope

This is a runtime graph rewrite. It does not modify the SavedModel on disk and does not rewrite the checkpoint files.

The pass runs on TensorFlow execution graphs created by Serving. In practice this means it may run on several graph instances for a single model lifecycle, such as restore/init graphs and the request callable graph. The request graph is usually the one with suffix `_2` in the current dumps.

Current source files:

- `tensorflow/tensorflow/compiler/jit/freeze_readonly_variables_pass.cc`
- `tensorflow/tensorflow/compiler/jit/freeze_readonly_variables_pass.h`
- `tensorflow/tensorflow/compiler/jit/jit_compilation_pass_registration.cc`
- `tensorflow/tensorflow/compiler/jit/BUILD`

Auxiliary analyzer:

- `scripts/analyze_freeze_candidates.py`

## Activation

The pass is controlled by an environment variable:

```bash
export TF_XLA_FREEZE_VARIABLES_CHECKPOINT=/path/to/model/1/variables/variables
```

The value is a TensorFlow checkpoint prefix, not a complete filename. For a SavedModel layout like this:

```text
model/1/
  saved_model.pb
  variables/
    variables.index
    variables.data-00000-of-00001
```

the prefix is:

```text
model/1/variables/variables
```

If the environment variable is not set or is empty, the pass returns without changing the graph.

In the current Serving launcher, the environment variable is set from the selected model path:

```bash
export TF_XLA_FREEZE_VARIABLES_CHECKPOINT="${MODEL}/1/variables/variables"
```

Graph dumps are controlled by `TF_DUMP_GRAPH_PREFIX`. The current launcher writes dumps under:

```text
tmp/tf_dump_graph/<MODEL_NAME_SHORT>/
```

## Pass Ordering

The pass is registered as a `POST_REWRITE_FOR_EXEC` optimization pass at priority `8`:

```cpp
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 8,
                      FreezeReadonlyVariablesPass);
```

Relevant ordering:

```text
priority 5   CloneConstantsForBetterClusteringPass
priority 8   FreezeReadonlyVariablesPass
priority 9   ClusterScopingPass
priority 10  MarkForCompilationPass
priority 12  ForceXlaConstantsOnHostPass
priority 20  IncreaseDynamismForAutoJitPass
```

This means freezing happens before XLA clustering decisions are made by `MarkForCompilationPass`.

## Checkpoint Model

The SavedModel graph describes variables and how they are used. The checkpoint stores the actual tensor values for those variables.

Example:

```text
GraphDef node:
  dnn/dnn_layers/hiddenlayer_0/kernel/part_0  op=VariableV2

Checkpoint key:
  dnn/dnn_layers/hiddenlayer_0/kernel         -> float tensor value
```

The pass uses `BundleReader` to open the checkpoint prefix, then reads dtype, shape, and tensor content for each matched key. The loaded tensor is serialized into a `TensorProto` and placed into a replacement `Const` node.

This is why `DT_RESOURCE` variables can be frozen: a `VarHandleOp` produces a resource handle in the graph, but the checkpoint key stores the backing tensor value, such as a `DT_FLOAT` embedding table.

## Candidate Selection

Only these node types are considered variable candidates:

```text
VarHandleOp
VariableV2
```

Nodes such as `_Arg`, `Placeholder`, request inputs, and normal compute ops are not candidates. For example, the current DFFM request graph has 39 `_Arg` nodes, but none of them enter the candidate set.

The candidate count is based on the current execution graph, not the original SavedModel. For DFFM, the original top-level graph has 105 variable nodes, but the request graph has 104 candidates because `global_step` is not used by the inference callable.

## Checkpoint Key Matching

For each candidate variable node, the pass tries keys derived from:

1. `shared_name`, if present and non-empty.
2. `node.name()`.

For each base name it also tries:

```text
<base>
<base>/.ATTRIBUTES/VARIABLE_VALUE
```

The current implementation also normalizes two naming patterns:

```text
varhandle/<name>  -> <name>
<name>/part_0     -> <name>
```

The `/part_0` suffix is a TensorFlow variable partition/shard naming pattern in the graph. The checkpoint key often stores the logical variable name without `/part_0`, even when there is only one shard.

The `unclustered/` prefix is not part of current key normalization. That prefix is created by `MarkForCompilationPass` for the annotated visualization dump and should not appear in the real pre-Mark freeze input graph.

## Safety Checks

The pass freezes only variables that look readonly in the current graph.

Any data edge from a variable to a mutating op makes the variable unsafe. Mutating ops include:

```text
Assign
AssignAdd
AssignSub
AssignVariableOp
AssignAddVariableOp
AssignSubVariableOp
DestroyResourceOp
ResourceApply*
ResourceScatter*
Scatter*
```

For `VarHandleOp`, each data consumer must be one of the supported readonly resource consumers:

```text
ReadVariableOp
ResourceGather
ResourceGatherNd
```

Additional checks:

- `ReadVariableOp` dtype must match the checkpoint tensor dtype.
- `ResourceGather` must have `batch_dims == 0` and matching output dtype.
- `ResourceGatherNd` output dtype must match the checkpoint tensor dtype.

For `VariableV2`, the pass rejects edges that feed ref-typed consumer inputs. This avoids freezing variables that are still participating in reference mutation paths.

Restore graphs usually have variable candidates connected to assign or restore logic, so they are intentionally skipped as unsafe. Request graphs usually contain readonly inference paths and are the intended target.

## Rewrite Rules

The rewrite happens in two phases.

First, variable storage nodes are replaced by constants loaded from checkpoint:

```text
VariableV2  -> Const
VarHandleOp -> Const
```

The replacement `Const` keeps the original node name, requested device, assigned device, control inputs, and internal attributes. Keeping the original name reduces downstream edge disruption.

Second, selected resource read/gather ops are rewritten into normal tensor ops:

```text
ReadVariableOp   -> Identity
ResourceGather   -> GatherV2
ResourceGatherNd -> GatherNd
```

### Why `ReadVariableOp` Becomes `Identity`

Before freezing, a resource variable read looks like this:

```text
VarHandleOp(DT_RESOURCE)
  -> ReadVariableOp(dtype=DT_FLOAT)
    -> consumer
```

After the variable node is replaced, the upstream node is already a tensor constant:

```text
Const(DT_FLOAT, checkpoint_value)
  -> ReadVariableOp
```

`ReadVariableOp` expects a `DT_RESOURCE` handle as input, so it cannot remain. Its remaining semantic role is just to forward the already-materialized tensor. Therefore it is replaced with:

```text
Const(DT_FLOAT, checkpoint_value)
  -> Identity(T=DT_FLOAT)
    -> consumer
```

This avoids duplicating large constants at every read site and keeps the original read node name and fanout stable.

### Why `ResourceGather` Becomes `GatherV2`

`ResourceGather` gathers from a resource variable. After freezing, the params input is a normal tensor constant, so the tensor equivalent is `GatherV2` with axis `0`.

The pass creates an extra scalar axis `Const` node and rewrites:

```text
ResourceGather(resource, indices) -> GatherV2(params, indices, axis=0)
```

Only `Tindices` values `DT_INT32` and `DT_INT64` are currently supported for the generated axis constant.

### Why `ResourceGatherNd` Becomes `GatherNd`

`ResourceGatherNd` gathers from a resource variable. After freezing, the params input is a normal tensor constant, so the tensor equivalent is:

```text
ResourceGatherNd(resource, indices) -> GatherNd(params, indices)
```

## Dumps And Logs

The pass writes before/after dumps:

```text
before_freeze_readonly_variables_pass*.pbtxt
after_freeze_readonly_variables_pass*.pbtxt
```

With the current Serving runs, useful request graph files are typically:

```text
tmp/tf_dump_graph/DFFM/before_freeze_readonly_variables_pass_2.pbtxt
tmp/tf_dump_graph/DFFM/after_freeze_readonly_variables_pass_2.pbtxt
tmp/tf_dump_graph/DLRM/before_freeze_readonly_variables_pass_2.pbtxt
tmp/tf_dump_graph/DLRM/after_freeze_readonly_variables_pass_2.pbtxt
```

The no-suffix files often correspond to restore graphs. Suffix `_1` and `_2` depend on graph creation order, but in the observed DFFM and DLRM Serving runs `_2` is the request callable graph.

Summary log line:

```text
FreezeReadonlyVariablesPass checkpoint=... candidates=... frozen_variables=...
  rewritten_reads=... rewritten_gathers=... rewritten_gather_nds=...
  skipped_missing_value=... skipped_unsafe=...
```

Per-variable log line:

```text
FreezeReadonlyVariablesPass freezing node=<node> op=<op>
  shared_name=<shared_name> checkpoint_key=<key> dtype=<dtype>
```

## Validation Script

Use the analyzer script to compare the original SavedModel, the checkpoint, and the before/after freeze request graphs.

For DFFM:

```bash
python scripts/analyze_freeze_candidates.py \
  --model-name DFFM \
  --model-dir models/model_DFFM/1731397983/1 \
  --list-limit 5
```

For DLRM:

```bash
python scripts/analyze_freeze_candidates.py \
  --model-name DLRM \
  --model-dir models/model_DLRM/1731305215/1 \
  --list-limit 5
```

The script reports:

```text
candidate variable nodes
candidate op counts
_Arg nodes
original top-level variable nodes
orig vars not present as request candidates
request candidates not in original top-level
candidate names ending /part_0
checkpoint keys ending /part_0
before/after _Arg changes
before/after VariableV2/VarHandleOp counts
after Const replacements
```

Expected current DFFM request-graph shape:

```text
candidate variable nodes: 104
candidate op counts: VarHandleOp=40, VariableV2=64
_Arg nodes: 39
original top-level variable nodes: 105
missing from request: global_step
before VariableV2/VarHandleOp count: 104
after VariableV2/VarHandleOp count: 0
```

Expected current DLRM request-graph shape:

```text
candidate variable nodes: 46
candidate op counts: VariableV2=46
_Arg nodes: 39
original top-level variable nodes: 57
before VariableV2/VarHandleOp count: 46
after VariableV2/VarHandleOp count: 0
```

## How To Read The Results

If restore graphs report candidates but `frozen_variables=0` and `skipped_unsafe` equals the candidate count, that is expected. Restore paths mutate variables and are not safe to freeze.

If the request graph reports candidates equal to frozen variables, then those variables were replaced by `Const` before XLA marking.

If `_Arg` count and names are unchanged before and after freeze, request inputs were not rewritten.

If checkpoint keys do not contain `/part_0` but graph candidates do, that is expected. The pass strips `/part_0` during key lookup.

If `mark_for_compilation_annotated_*.pbtxt` contains `unclustered/`, that is from the MarkForCompilation visualization copy. Use `before_freeze`, `after_freeze`, or `before_mark_for_compilation` dumps when judging the real graph entering XLA clustering.

## Current Limitations

The implementation is intentionally narrow and suitable for quick validation rather than a general TensorFlow freezing pass.

Known limits:

- Only top-level `Graph` nodes are processed.
- FunctionDef bodies and captured resources inside functions are not rewritten.
- Only `VariableV2` and `VarHandleOp` are candidates.
- Only `ReadVariableOp`, `ResourceGather`, and `ResourceGatherNd` resource consumers are rewritten.
- `ResourceGather` supports only `batch_dims == 0`.
- Generated `GatherV2` axis constants support only `DT_INT32` and `DT_INT64` indices dtype.
- Variables connected to mutation/reference paths are skipped.
- The pass reads checkpoint values at graph optimization time, so graph size can grow when large variables are embedded as `Const` TensorProtos.

## Mental Model

Think of the pass as replacing this runtime pattern:

```text
checkpoint value loaded into variable storage
request graph reads variable storage
XLA sees variable/resource operations
```

with this request-graph pattern:

```text
checkpoint value embedded directly into Const
request graph uses normal tensor ops
XLA sees constants and tensor computation
```

The checkpoint remains the source of truth for values. The SavedModel remains unchanged. Only the in-memory execution graph is rewritten before XLA clustering.