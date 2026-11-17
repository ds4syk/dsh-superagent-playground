import FourColor

/-!
# 独立审计：四色定理形式化的最终定理

这个文件不依赖上游的 `scripts/Check.lean`，而是直接用 Lean 的环境 API
检查以下四件事：

1. `FourColor.fourColorTheorem` 存在；
2. 它的类型恰好是 `FourColor.FourColorTheorem`（而不是某个更弱的命题）；
3. 该命题按定义展开后就是“每个简单平面地图四色可着色”；
4. 它的公理依赖不超过 `propext`、`Classical.choice`、`Quot.sound`，
   并且 `FourColor` 命名空间中没有声明依赖 `sorryAx`。

运行方式（在上游仓库根目录）：

```shell
lake env lean /path/to/Audit.lean
```

成功时最后一行是 `AUDIT OK: ...`。
-/

open Lean Elab Command

namespace FourColorAudit

/-- 定理可以直接给出所声明的命题。 -/
example : FourColor.FourColorTheorem := FourColor.fourColorTheorem

/-- `FourColorTheorem` 按定义就是地图版本的四色定理。 -/
example : FourColor.FourColorTheorem =
    (∀ m : FourColor.PlaneMap, FourColor.SimpleMap m → FourColor.ColorableWith 4 m) := rfl

#check FourColor.fourColorTheorem
#print FourColor.FourColorTheorem
#print axioms FourColor.fourColorTheorem

run_cmd do
  let env ← getEnv
  let goal : Name := `FourColor.fourColorTheorem
  let goalType : Name := `FourColor.FourColorTheorem
  let some info := env.find? goal
    | throwError "AUDIT FAIL: {goal} 不在环境中"
  unless info.type.isConstOf goalType do
    throwError "AUDIT FAIL: {goal} 的类型是 {info.type}，不是 {goalType}"
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let axs ← liftCoreM <| collectAxioms goal
  let extra := axs.filter (fun a => !allowed.contains a)
  unless extra.isEmpty do
    throwError "AUDIT FAIL: 非标准公理依赖 {extra.toList}"
  let mut declarations := 0
  let mut sorryDeclarations := 0
  for (c, _) in env.constants.toList do
    if (`FourColor).isPrefixOf c && !c.isInternal then
      declarations := declarations + 1
      for a in (← liftCoreM <| collectAxioms c) do
        if a == ``sorryAx then
          sorryDeclarations := sorryDeclarations + 1
  unless sorryDeclarations == 0 do
    throwError "AUDIT FAIL: {sorryDeclarations} 个声明依赖 sorryAx"
  logInfo s!"AUDIT OK: {goal} : {goalType}; \
axioms = {axs.toList}; declarations = {declarations}; sorryAx = 0"

end FourColorAudit
