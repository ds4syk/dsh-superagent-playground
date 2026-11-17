# Lean 4 检查：μ(π) ≥ 2（本报告 S1 + S2）

本目录形式化本报告已证明的**下界**部分：

- `GoodApproximations` / `ApproximationExponents` 定义（对应 §1.1）；
- `finite_den_one_good`：分母为 1 的好逼近只有有限多个；
- `infinite_den_ge_two_sq`：分母 ≥ 2、误差 < 1/q.den² 的逼近无穷多；
- `mu_pi_ge_two`：`2 ∈ ApproximationExponents π`，即 μ(π) ≥ 2；
- `mu_pi_ge_two_of_lt`：`∀ ν < 2` 的无穷多逼近形式。

**不包含**：μ(π) ≤ 2（未证/官方声称）、连分数公式 S3、双见证 S5。

## 验证

本机使用已有的 Lean/Mathlib 缓存：

```bash
cd /mnt/storage1/syk/tmp-notebook-work/fourcolor-audit/a
export PATH=$HOME/.elan/bin:$PATH
lake env lean /mnt/storage1/syk/.superagent-work/pi-exponent-proof-dd8280cf5917/lean/PiExponent.lean
```

- Lean 工具链：`leanprover/lean4:v4.34.0-rc2`（四色项目提供；官方 openai/math 用 v4.34.1，本检查不依赖官方项目）。
- 无 `sorry`、`admit` 或自定义 `axiom`。
- 因为 `sSup` 的下界需要指数集合有上界（即未证的 μ(π) < ∞），这里不定义 `μ := sSup ...` 再证 `2 ≤ μ`；形式化的是等价下界命题 `2 ∈ ApproximationExponents π`，并在文件头注明。

## 检查结果（2026-11-17）

- `rc=0`，无 `sorry`/`admit`/自定义 `axiom`。
- `#print axioms`：
  - `PiExponent.mu_pi_ge_two`：`[propext, Classical.choice, Quot.sound]`
  - `PiExponent.mu_pi_ge_two_of_lt`：`[propext, Classical.choice, Quot.sound]`
- 完整输出见 [`check.log`](check.log)。
