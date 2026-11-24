import Mathlib

/-!
# A machine-checked 1-D tie / no-free-lunch lemma

For `J x θ = (x^2 - 1)^2 + θ x`, the global minimizer is negative for `θ > 0`
and positive for `θ < 0`.  Hence no continuous selection of the global
minimizer can take either value `±1` at `θ = 0`; the failure of continuity of
an exact selection at a tie is already visible in one dimension.

This file is the Lean-checked core of the 1-D tie lower bound.  The general
face--stratum theorem is proved in the paper; a full Lean formalization of the
SDP/KKT/IFT part is future work.
-/

namespace FaceStratum

noncomputable section
open Filter Set
open scoped Topology

/-- The 1-D quartic family. -/
def J (x θ : ℝ) : ℝ := (x ^ 2 - 1) ^ 2 + θ * x

lemma J_neg_one (θ : ℝ) : J (-1) θ = -θ := by
  unfold J
  ring

lemma J_one (θ : ℝ) : J 1 θ = θ := by
  unfold J
  ring

/-- For `θ > 0` every global minimizer is negative. -/
lemma minimizer_neg {θ x : ℝ} (hθ : 0 < θ) (hx : ∀ y, J x θ ≤ J y θ) : x < 0 := by
  by_contra h
  push_neg at h
  have h1 : J x θ ≤ J (-1) θ := hx (-1)
  rw [J_neg_one] at h1
  have h2 : 0 < J x θ - (-θ) := by
    have h3 : 0 ≤ (x ^ 2 - 1) ^ 2 := sq_nonneg _
    have h4 : 0 < θ * (x + 1) := mul_pos hθ (by linarith)
    have h5 : J x θ - (-θ) = (x ^ 2 - 1) ^ 2 + θ * (x + 1) := by
      simp [J]
      ring
    rw [h5]
    nlinarith
  linarith

/-- For `θ < 0` every global minimizer is positive. -/
lemma minimizer_pos {θ x : ℝ} (hθ : θ < 0) (hx : ∀ y, J x θ ≤ J y θ) : 0 < x := by
  by_contra h
  push_neg at h
  have h1 : J x θ ≤ J 1 θ := hx 1
  rw [J_one] at h1
  have h2 : 0 < J x θ - θ := by
    have h3 : 0 ≤ (x ^ 2 - 1) ^ 2 := sq_nonneg _
    have h4 : 0 < θ * (x - 1) := mul_pos_of_neg_of_neg hθ (by linarith)
    have h5 : J x θ - θ = (x ^ 2 - 1) ^ 2 + θ * (x - 1) := by
      simp [J]
      ring
    rw [h5]
    nlinarith
  linarith

/-- Continuity at `0` with value `1` gives positivity nearby. -/
lemma eventually_pos_of_continuousAt {s : ℝ → ℝ} (h : ContinuousAt s 0) (h0 : s 0 = 1) :
    ∀ᶠ θ in 𝓝 (0 : ℝ), 0 < s θ := by
  have hdist : ∀ᶠ θ in 𝓝 (0 : ℝ), dist (s θ) 1 < (1 / 2 : ℝ) := by
    rw [Metric.eventually_nhds_iff]
    simpa [Real.dist_eq, h0] using (Metric.continuousAt_iff.mp h) (1 / 2) (by norm_num)
  filter_upwards [hdist] with θ hθ
  rw [Real.dist_eq] at hθ
  have h1 : -(1 / 2 : ℝ) < s θ - 1 := (abs_lt.mp hθ).1
  linarith

/-- Continuity at `0` with value `-1` gives negativity nearby. -/
lemma eventually_neg_of_continuousAt {s : ℝ → ℝ} (h : ContinuousAt s 0) (h0 : s 0 = -1) :
    ∀ᶠ θ in 𝓝 (0 : ℝ), s θ < 0 := by
  have hdist : ∀ᶠ θ in 𝓝 (0 : ℝ), dist (s θ) (-1) < (1 / 2 : ℝ) := by
    rw [Metric.eventually_nhds_iff]
    simpa [Real.dist_eq, h0] using (Metric.continuousAt_iff.mp h) (1 / 2) (by norm_num)
  filter_upwards [hdist] with θ hθ
  rw [Real.dist_eq] at hθ
  have h1 : s θ - (-1) < (1 / 2 : ℝ) := (abs_lt.mp hθ).2
  linarith

/-- An eventually positive property holds at some positive point. -/
lemma exists_pos_of_eventually_pos {p : ℝ → Prop} (h : ∀ᶠ θ in 𝓝 (0 : ℝ), p θ) :
    ∃ θ > 0, p θ := by
  rw [Metric.eventually_nhds_iff] at h
  rcases h with ⟨ε, hεpos, hε⟩
  refine ⟨ε / 2, by linarith, hε ?_⟩
  rw [Real.dist_eq, sub_zero, abs_of_pos (by linarith : (0 : ℝ) < ε / 2)]
  linarith

/-- An eventually negative property holds at some negative point. -/
lemma exists_neg_of_eventually_neg {p : ℝ → Prop} (h : ∀ᶠ θ in 𝓝 (0 : ℝ), p θ) :
    ∃ θ < 0, p θ := by
  rw [Metric.eventually_nhds_iff] at h
  rcases h with ⟨ε, hεpos, hε⟩
  refine ⟨-(ε / 2), by linarith, hε ?_⟩
  rw [Real.dist_eq]
  have : -(ε / 2) - 0 = -(ε / 2) := by ring
  rw [this, abs_neg, abs_of_pos (by linarith : (0 : ℝ) < ε / 2)]
  linarith

/--
**No continuous selection at the 1-D tie.**
There is no continuous selection of the global minimizer of `J(·, θ)` that
takes the value `1` or `-1` at `θ = 0`.
-/
theorem no_continuous_selection :
    ¬ ∃ s : ℝ → ℝ, ContinuousAt s 0 ∧ (s 0 = 1 ∨ s 0 = -1) ∧
      ∀ θ, ∀ y, J (s θ) θ ≤ J y θ := by
  rintro ⟨s, hcont, h0, hmin⟩
  rcases h0 with h0 | h0
  · have hpos := eventually_pos_of_continuousAt hcont h0
    rcases exists_pos_of_eventually_pos hpos with ⟨θ, hθ, hs⟩
    have hn : s θ < 0 := minimizer_neg hθ (hmin θ)
    linarith
  · have hneg := eventually_neg_of_continuousAt hcont h0
    rcases exists_neg_of_eventually_neg hneg with ⟨θ, hθ, hs⟩
    have hp : 0 < s θ := minimizer_pos hθ (hmin θ)
    linarith

end
end FaceStratum
