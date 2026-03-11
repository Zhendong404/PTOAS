1. CreateFusionGroupPass放到InsertSync之前，让Sync感知融合，选择最佳插入位置
2. 融合后load/store消除
3. Low-level IR库方案端到端打通（IR设计、PTO-ISA库转换、EmitC、测试）
4. 自动识别vec_scope,避免oplib开发者手动插入
5. vec_scope缩进不太美观
6. for循环没有使用size_t
(x) 7. oplib的row/col是静态的
(x) 8. oplib只有1D版本
(x) 9. oplib里没有实现tile_buf to memref的转换
(x) 10. pto.simd里为tile_buf -> memref的转换定义一个IR（当前是unrealized_conversion_cast），legacy tile_buf-to-memref bridge stage/View2Memref转换时处理掉
(x) 11. 只对A5使用新的oplib codegen路线
(不做)12. 重构plan memory和insert sync到TileBuf world
13. vector lane的数量是写死的64
(x) 14. 重构pto.simd.tile_to_memref的lowering逻辑，变成一个Tile.data()
15. 把所有用例打包成1个BIN文件上板，提高验证效率