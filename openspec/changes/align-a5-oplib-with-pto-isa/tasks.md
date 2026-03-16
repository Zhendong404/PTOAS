## 1. Manifest 对齐

- [x] 1.1 审查并修正 `oplib/level3/families/a5_oplib_v1_manifest.yaml` 中已确认失真的条目，至少覆盖 `trecip`、bitwise family 的 `dtype_support` 和异常 `key_constraints`
- [x] 1.2 更新 manifest 生成或校验逻辑，使其能识别同级 `pto-isa/include/pto/common/pto_instr.hpp` 的公共 API 语义和 `tests/npu/a5/src/st/testcase/` 的 A5 ST 证据
- [x] 1.3 为 `implemented` / `deferred` 分类补充结构化校验，明确区分“原生 A5 `_IMPL`”、“公共 API 语义重写”和“仍缺少可采纳语义”

## 2. Family 模板补齐

- [x] 2.1 扩展 compare/select family 的 generator / template 轴，补齐 `tcmp`、`tcmps`、`tsel`、`tsels` 的非 `f32` dtype 覆盖
- [x] 2.2 扩展 reduction、broadcast 和 scalar-expand family，至少补齐 `trowsum` / `trowmax` / `trowmin` 的 `f16`，以及 `tcolsum` / `tcolmax` / `tcolmin`、`trowexpand` / `tcolexpand`、`texpands` 的 A5 真实 dtype 集
- [ ] 2.3 扩展 arithmetic / tile-scalar / bitwise family 的 `bf16`、unsigned 和缺失的整型 variant，并确保模板 metadata 不再声明 A5 不支持的 dtype
- [ ] 2.4 评估 `trecip` 的 OpLib 接入方式，若采用公共 API 等价语义，补齐对应 template / matcher / metadata；对 ternary 四个 OP 保持显式 deferred

## 3. Lowering 与一致性门禁

- [ ] 3.1 调整 `PTOLowerToOpLibCalls` 相关逻辑，使 manifest 状态、template 选择和 deferred 诊断与新分类规则一致
- [ ] 3.2 升级 `test/oplib/check_implemented_op_alignment.py` 或新增检查脚本，使其能校验 implemented op 的 dtype 级覆盖，而不只是“至少一个模板”
- [ ] 3.3 补充对 `trecip`、compare/select、reduction、broadcast、scalar-expand、bitwise 关键 family 的 lowering use case，确保新 dtype / variant 能进入 OP-Lib path

## 4. 回归、文档与收尾

- [ ] 4.1 补充或更新 `test/oplib/` lit 回归，覆盖 manifest 纠偏、family dtype 扩展、deferred 诊断和 `trecip` 对齐行为
- [ ] 4.2 更新 `docs/tile_fusion/a5_oplib_v1_authoring.md` 及相关文档，说明新的语义证据来源、implemented/deferred 规则和 decomposition 边界
- [ ] 4.3 运行并记录本 change 相关验证，包括 generator 校验、implemented-op 对齐检查，以及聚焦 `test/oplib` 的 lit 回归
