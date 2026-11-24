import Mathlib

/-!
# Machine-checked rank component of value--face separation

For real `n × n` matrices, if `W * M = 0`, then the range of `M` is contained
in the kernel of `W`, hence `rank M ≤ dim ker W`.  The remaining PSD step
(`W, M ⪰ 0` and `tr (W M) = 0` imply `W * M = 0`) is proved in the paper via
the spectral theorem; it is not claimed as Lean-checked here.
-/

namespace FaceStratum

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- If `W * M = 0`, then `range M ⊆ ker W`. -/
theorem range_mulVec_le_ker {W M : Matrix n n ℝ} (h : W * M = 0) :
    LinearMap.range M.mulVecLin ≤ LinearMap.ker W.mulVecLin := by
  rw [LinearMap.range_le_ker_iff]
  ext x i
  have hx : (W * M) i x = 0 := by rw [h]; rfl
  simpa [Matrix.mulVec, dotProduct, Matrix.mul_apply] using hx

/-- Rank bound: `rank M ≤ dim ker W` when `W * M = 0`. -/
theorem rank_le_finrank_ker {W M : Matrix n n ℝ} (h : W * M = 0) :
    M.rank ≤ Module.finrank ℝ (LinearMap.ker W.mulVecLin) := by
  have hrange := range_mulVec_le_ker h
  calc
    M.rank = Module.finrank ℝ (LinearMap.range M.mulVecLin) := by
      rw [Matrix.rank_eq_finrank_range_toLin M (Pi.basisFun ℝ n) (Pi.basisFun ℝ n)]
      rfl
    _ ≤ Module.finrank ℝ (LinearMap.ker W.mulVecLin) := Submodule.finrank_mono hrange

end FaceStratum
