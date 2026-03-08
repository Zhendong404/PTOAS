1. CreateFusionGroupPass放到InsertSync之前，让Sync感知融合，选择最佳插入位置
2. 融合后load/store消除
3. Low-level IR库方案端到端打通（IR设计、PTO-ISA库转换、EmitC、测试）
4. 自动识别vec_scope,避免oplib开发者手动插入
5. vec_scope缩进不太美观
6. for循环没有使用size_t
7. oplib的row/col是静态的
8. oplib只有1D版本
9. oplib里没有实现tile_buf to memref的转换
10. pto.simd里为tile_buf -> memref的转换定义一个IR（当前是unrealized_conversion_cast），Tilebuf2Memref/View2Memref转换时处理掉