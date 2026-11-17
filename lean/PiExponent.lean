/-
  π 的无理性指数：下界 μ(π) ≥ 2 的 Lean 4 检查。

  本文件只形式化本报告已证明的部分（S1 定义 + S2 下界），不包含 μ(π) ≤ 2
  （公开/官方声称的难点）与连分数公式 S3。

  数学依据：
  * Mathlib: `Real.infinite_rat_abs_sub_lt_one_div_den_sq_of_irrational`
    （无理数有无穷多个 |x - q| < 1/q.den² 的有理逼近）；
  * Mathlib: `irrational_pi`（π 无理性）；
  * 去掉分母为 1 的有限多个逼近，得到分母 ≥ 2 的无穷多逼近；
  * 对 ν < 2，把 q^{-2} 放大为 q^{-ν}。

  注意：这里不把 μ 定义为 `sSup` 再证 `2 ≤ sSup`，因为那需要指数集合有上界
  （即 μ(π) < ∞，正是未证的难点）。这里形式化的是等价的下界命题：
  `2 ∈ ApproximationExponents π`（以及 `∀ ν < 2` 的无穷多逼近形式）。
-/
import Mathlib.Analysis.Real.Pi.Irrational
import Mathlib.NumberTheory.DiophantineApproximation.Basic

open Set

namespace PiExponent

/-- 分母至少 2、误差小于 q^{-ν} 的有理逼近集合。 -/
def GoodApproximations (x : ℝ) (ν : ℝ) : Set ℚ :=
  {q | 2 ≤ q.den ∧ 0 < |x - (q : ℝ)| ∧ |x - (q : ℝ)| < (q.den : ℝ) ^ (-ν)}

/-- 使“有无穷多个好逼近”成立的指数 ν 的集合。 -/
def ApproximationExponents (x : ℝ) : Set ℝ :=
  {ν | 0 < ν ∧ (GoodApproximations x ν).Infinite}

/-- 分母为 1 的好逼近只有有限多个：它们是落在 (π-1, π+1) 内的整数。 -/
theorem finite_den_one_good :
    {q : ℚ | q.den = 1 ∧ |Real.pi - (q : ℝ)| < 1 / (q.den : ℝ) ^ 2}.Finite := by
  have hInt : {n : ℤ | |Real.pi - (n : ℝ)| < 1}.Finite := by
    refine Set.Finite.subset (Set.finite_Icc (⌊Real.pi⌋) (⌈Real.pi⌉)) ?_
    intro n hn
    change |Real.pi - (n : ℝ)| < 1 at hn
    change ⌊Real.pi⌋ ≤ n ∧ n ≤ ⌈Real.pi⌉
    constructor
    · by_contra h
      have hlt : n < ⌊Real.pi⌋ := by omega
      have hIntLe : n ≤ ⌊Real.pi⌋ - 1 := by omega
      have h1 : (n : ℝ) ≤ (⌊Real.pi⌋ : ℝ) - 1 := by exact_mod_cast hIntLe
      have h2 : (⌊Real.pi⌋ : ℝ) ≤ Real.pi := Int.floor_le _
      have h3 : 1 ≤ Real.pi - (n : ℝ) := by linarith
      have h4 : 1 ≤ |Real.pi - (n : ℝ)| := by
        rw [abs_of_nonneg (by linarith)]
        linarith
      linarith
    · by_contra h
      have hlt : ⌈Real.pi⌉ < n := by omega
      have hIntLe : ⌈Real.pi⌉ + 1 ≤ n := by omega
      have h1 : (⌈Real.pi⌉ : ℝ) + 1 ≤ (n : ℝ) := by exact_mod_cast hIntLe
      have h2 : Real.pi ≤ (⌈Real.pi⌉ : ℝ) := Int.le_ceil _
      have h3 : 1 ≤ (n : ℝ) - Real.pi := by linarith
      have h4 : 1 ≤ |Real.pi - (n : ℝ)| := by
        rw [abs_of_nonpos (by linarith)]
        linarith
      linarith
  refine Set.Finite.subset (hInt.image (fun n : ℤ => (n : ℚ))) ?_
  intro q hq
  rcases hq with ⟨hden, herr⟩
  change ∃ n : ℤ, |Real.pi - (n : ℝ)| < 1 ∧ (n : ℚ) = q
  refine ⟨q.num, ?_, ?_⟩
  · have hqeq : (q.num : ℚ) = q := (Rat.den_eq_one_iff q).mp hden
    have hcast : (q.num : ℝ) = (q : ℝ) := by
      rw [← Rat.cast_intCast, hqeq]
    simpa [hden, hcast] using herr
  · exact (Rat.den_eq_one_iff q).mp hden

/-- 分母至少 2、误差 < 1/q.den² 的有理逼近有无穷多个。 -/
theorem infinite_den_ge_two_sq :
    {q : ℚ | 2 ≤ q.den ∧ |Real.pi - (q : ℝ)| < 1 / (q.den : ℝ) ^ 2}.Infinite := by
  have hInf : {q : ℚ | |Real.pi - (q : ℝ)| < 1 / (q.den : ℝ) ^ 2}.Infinite :=
    Real.infinite_rat_abs_sub_lt_one_div_den_sq_of_irrational irrational_pi
  have hdiff := hInf.sdiff finite_den_one_good
  refine hdiff.mono ?_
  intro q hq
  rcases hq with ⟨hgood, hnot⟩
  refine ⟨?_, hgood⟩
  by_contra hlt
  have hlt' : q.den < 2 := Nat.lt_of_not_le hlt
  have hpos : 0 < q.den := q.pos
  have hne : q.den ≠ 1 := by
    intro h
    exact hnot ⟨h, hgood⟩
  omega

/-- 形式化的 μ(π) ≥ 2：`2` 属于近似指数集合。 -/
theorem mu_pi_ge_two : (2 : ℝ) ∈ ApproximationExponents Real.pi := by
  refine ⟨by norm_num, ?_⟩
  refine infinite_den_ge_two_sq.mono ?_
  intro q hq
  rcases hq with ⟨hden, herr⟩
  refine ⟨hden, ?_, ?_⟩
  · exact abs_pos.mpr (sub_ne_zero.mpr (Irrational.ne_rat irrational_pi q))
  · -- 1 / q.den^2 = q.den^(-2)
    have hconv : (q.den : ℝ) ^ (-(2 : ℝ)) = 1 / (q.den : ℝ) ^ 2 := by
      rw [Real.rpow_neg (by positivity)]
      simp [one_div]
    rwa [hconv]

/-- 等价形式：对每个 ν < 2，都有无穷多个分母 ≥ 2、误差 < q^{-ν} 的逼近。 -/
theorem mu_pi_ge_two_of_lt (ν : ℝ) (hν : ν < 2) :
    (GoodApproximations Real.pi ν).Infinite := by
  refine mu_pi_ge_two.2.mono ?_
  intro q hq
  rcases hq with ⟨hden, hpos, herr⟩
  refine ⟨hden, hpos, ?_⟩
  have hbase : (1 : ℝ) < (q.den : ℝ) := by
    have : 1 < q.den := by omega
    exact_mod_cast this
  exact herr.trans (Real.rpow_lt_rpow_of_exponent_lt hbase (by linarith : (-2 : ℝ) < -ν))

end PiExponent

-- 公理审计：只应出现 Lean/Mathlib 的标准公理（propext、Classical.choice、Quot.sound）。
#print axioms PiExponent.mu_pi_ge_two
#print axioms PiExponent.mu_pi_ge_two_of_lt
