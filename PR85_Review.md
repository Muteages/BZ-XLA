# PR 85 Review

## Background

Target commit:

https://github.com/joeyye-work/tensorflow/pull/85

- `6ecf767dbb737dd930fd75aaced5f883c0636cff`
- Subject: `Move DynExpr* ownership into DExpr and use unique pointers (#85)`

Reference fix commit:

- `8662606077dce70d36282446fba2340f3ed0630c`
- Subject: `Fix accidental dynamic-shape regressions in DExpr ownership cleanup (#89)`

This note only focuses on changes in `6ecf767dbb7` that are suspicious because
they are not simple ownership migration from raw pointers to `unique_ptr` / `DExpr`.

Review rule:

- `建议回滚`: clearly unrelated behavior change, high probability accidental
- `需要单测验证`: possibly intentional cleanup, but it changes expr semantics or feature gating
- `可保留`: not mechanical, but low-risk formatting or representation cleanup

## Summary

The suspicious changes fall into three groups:

1. Regressions later fixed by `8662606077d`
2. Additional expr semantic changes not covered by `8662606077d`
3. Low-risk formatting and debug-output changes

## Table

| File | Suspicious change in `6ecf767dbb7` | Recommendation | Covered by `8662606077d` | Notes |
|---|---|---|---|---|
| `tensorflow/compiler/tf2xla/kernels/dynamic_partition_op.cc` | Replaced `Min(GetDimensionSize(data), GetDimensionSize(partitions))` with `GetDimensionSize(data)` only | 建议回滚 | Yes | Changes padding-mask behavior for mismatched dynamic information |
| `third_party/xla/xla/hlo/builder/lib/broadcast.cc` | Removed `output_exprs` vs `output_dims` rank check | 建议回滚 | Yes | Removes an input validity guard and can lead to invalid reverse-iterator usage |
| `third_party/xla/xla/hlo/builder/lib/broadcast.cc` | Removed branch that treats `output_dim == input_dim` or `input_dim == 1` as pure broadcast when deriving exprs | 建议回滚 | Yes | Changes broadcast expr semantics, especially for bound-1 dims with non-constant exprs |
| `third_party/xla/xla/hlo/builder/xla_builder.cc` | Changed `GetDimensionSize` dynamic path from frontend attributes to `Broadcast + GetExpressionValue` | 建议回滚 | Yes | Changes dynamic expr materialization path |
| `third_party/xla/xla/service/cpu/cpu_instruction_fusion.cc` | Removed protection that forbids CPU fusion for dynamic dots | 建议回滚 | Yes | Changes optimization behavior, not ownership |
| `tensorflow/core/grappler/costs/graph_properties.cc` | Replaced selective reshape expr refresh with unconditional `force_fresh_unknown_dim` for all reshape exprs | 建议回滚 | No | Old code only refreshed non-variable exprs on `Reshape`; new code refreshes all existing exprs |
| `tensorflow/compiler/tf2xla/ops/xla_ops.cc` | Removed `tf_xla_enable_dynamic_sizes` gate and always builds / attaches exprs | 需要单测验证 | No | Changes feature-flag behavior |
| `tensorflow/compiler/tf2xla/shape_util.cc` | Removed multiple `tf_xla_enable_dynamic_sizes` gates and always propagates exprs between `TensorShape` and `xla::Shape` | 需要单测验证 | No | Same behavior expansion as above |
| `tensorflow/core/common_runtime/constant_folding.cc` | Replaced `TensorShape(...).get_expressions()` scan with `HasDynamicDimExprs(proto)` | 需要单测验证 | No | Changes what counts as dynamic shape metadata |
| `tensorflow/core/framework/tensor_shape_expr.cc` | Added `IsDynamicDimExpr` and `HasDynamicDimExprs` | 需要单测验证 | No | New dynamic-expr detection rule, not ownership plumbing |
| `tensorflow/core/framework/tensor_shape.cc` | `NODE_TYPE_NOT_SET` now maps to `DExpr::Unknown()` instead of empty / null | 建议回滚 | No | Changes missing-expr semantics |
| `tensorflow/core/framework/tensor_shape.cc` | `set_expression`, `AddExpression`, `set_expressions` normalize empty exprs to `Unknown()` | 建议回滚 | No | Merges `missing expr` and `unknown expr` states |
| `tensorflow/core/framework/tensor_shape.cc` | `set_dim` now rewrites expr whenever expr vector length is sufficient | 建议回滚 | No | Previously only rewrote when an actual expr already existed |
| `tensorflow/core/framework/tensor_shape.cc` | `SetDimWithStatus` changed in the same way as `set_dim` | 建议回滚 | No | Same semantic widening as above |
| `tensorflow/core/framework/tensor_shape.h` | `get_expression()` missing case returns `kMissingExpression = DExpr::Unknown()` | 建议回滚 | No | Missing and unknown become conflated |
| `tensorflow/core/framework/tensor_shape.h` | `get_filled_expression()` returns `Unknown()` in invalid / missing cases | 建议回滚 | No | Same missing-vs-unknown semantic merge |
| `third_party/xla/xla/shape_dynexpr.h` | Introduced `UnknownExpr`, `DExpr::Unknown(id)`, and `operator bool()` that treats unknown as false | 建议回滚 | No | Ownership migration did not require introducing a new expr state |
| `third_party/xla/xla/shape_dynexpr.h` | `NODE_TYPE_NOT_SET` now returns `DExpr::Unknown()` | 建议回滚 | No | Changes proto decoding semantics |
| `third_party/xla/xla/shape.cc` | Introduced `Shape::MissingExpression()` backed by static `Unknown()` | 建议回滚 | No | Missing expr no longer behaves like empty |
| `third_party/xla/xla/shape.cc` | `DynExpr::equal` adds `UnknownExpr` equality by id | 需要单测验证 | No | New equality semantics for unknown exprs |
| `third_party/xla/xla/shape.cc` | `Mul::s`, `Add::s`, `Sub::s`, `Div::s` propagate `Unknown()` when either side is unknown | 需要单测验证 | No | Simplification semantics changed beyond ownership |
| `tensorflow/compiler/tf2xla/kernels/reshape_op.cc` | Placeholder expr for dynamic output dims changed from sentinel constant to `DExpr::Unknown(111)` | 需要单测验证 | No | Changes downstream interpretation from constant placeholder to unknown expr state |
| `third_party/xla/xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.cc` | Placeholder exprs changed from sentinel constants to `Unknown(40)` and `Unknown(50)` | 需要单测验证 | No | Same placeholder semantic change |
| `third_party/xla/xla/hlo/translate/mhlo_to_hlo/type_to_shape.cc` | Default exprs changed from sentinel constants to `Unknown(60)` | 需要单测验证 | No | Same placeholder semantic change |
| `third_party/xla/xla/service/shape_inference.cc` | Some default expr placeholders changed to `Unknown(70)` / `Unknown(80)` | 需要单测验证 | No | Same placeholder semantic change |
| `tensorflow/core/framework/tensor_shape.cc` | `ExprToString` moved from custom formatting to `StringPrinter` | 可保留 | No | Debug / formatting only |
| `third_party/xla/xla/shape.cc` | `operator<<` for expr moved from proto `ShortDebugString` to `StringPrinter` | 可保留 | No | Debug / formatting only |

## Most Important Unfixed Items

If only a short list is needed, the highest-priority suspicious changes that
were not fixed by `8662606077d` are:

1. `tensorflow/core/grappler/costs/graph_properties.cc`
2. `tensorflow/core/framework/tensor_shape.cc`
3. `tensorflow/core/framework/tensor_shape.h`
4. `third_party/xla/xla/shape_dynexpr.h`
5. `third_party/xla/xla/shape.cc`
6. `tensorflow/compiler/tf2xla/ops/xla_ops.cc`
7. `tensorflow/compiler/tf2xla/shape_util.cc`

## Why `graph_properties.cc` Is Suspicious

Original logic in `6ecf767dbb7^`:

- On `Reshape`, refresh expr only when an existing expr is present and it is
  not a plain `Variable`
- Keep existing variable exprs stable

Logic after `6ecf767dbb7`:

- On `Reshape`, if a dim already has any expr at all, always replace it with a
  fresh unknown canonical dim

This is not required by `unique_ptr` migration. It changes the symbolic-shape
canonicalization policy for reshape outputs.

## Suggested Next Review Steps

1. Re-check whether `UnknownExpr` introduction was intentional in a separate design note or only appeared inside this ownership commit.
2. Add targeted tests around `Reshape`, `TensorShape::set_dim`, and shape-proto decoding with `NODE_TYPE_NOT_SET`.
3. If a minimal revert is desired, first revert the `graph_properties.cc`, `tensor_shape.*`, and `shape*.{cc,h}` semantic deltas while keeping the ownership migration intact.