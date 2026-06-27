1. 背景：PTOAS支持mix kernel写法，但依赖pto.section.cube/vector region将对应的section标注出来。PTODSL希望将mix kernel模式作为主推的编程风格，但不引入cube/vector section概念，从而降低用户心智负担（判断一个OP属于cube/vector section要付出额外的心智）
2. 现有方案：PTODSL直接生成不含section信息的IR，PTOAS通过一个自动section推断pass将cube/vector section的OP自动封装起来。对于EmitC后端，section region会转换为#if defined(__DAV_CUBE__) / #if defined(__DAV_VEC__)宏，在CCE编译器中进行split然后编译；对于VPTO后端，PTOAS会在VPTO后端流水线中通过CVSplitModule pass将cube/vector section分成2个不同的module，分别进行编译；
3. 问题：对于使用CV pipe的场景，aic_initialize_pipe/aiv_initialize_pipe OP要求不能处于同一个函数中，这意味着mix kernel中无法使用CV pipe
4. 解决方案1：将CVSplitModule pass作为2个后端的公共pass，放到PTOLowerFrontendPipeOpsPass之前，将CV分离成2个不同的函数，使PTOLowerFrontendPipeOpsPass正常工作。风险：过早分离CV，可能导致plan memory/insert sync pass失效
5. 解决方案2：修改PTOLowerFrontendPipeOpsPass，使其支持在同一函数的不同section中工作，在后面再进行CV Split。风险：CV pipe的flag id分配等算法可能需要调整
6. 解决方案3：约束mix kernel不能使用CV pipe；要使用CV pipe必须通过CV split写法