Minimum commands to run a single standalone st


```
# 0) env
cd /workdir/ptoas_a5
source set_ptoas_env.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH}"

# 1) build (tadd currently fails here; tload succeeds)
ST=/workdir/ptoas_a5/test/tilelang_st/npu/a5/src/st
cd "$ST" && rm -rf build && mkdir build && cd build
cmake .. -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 -DTEST_CASE=tadd \
  -DPTOAS_BIN=/workdir/ptoas_a5/build/tools/ptoas/ptoas
make -j"$(nproc)" tadd          # ← ✅  works now with beta1

export LD_LIBRARY_PATH="${ST}/build/lib:${LD_LIBRARY_PATH}"

# 2) gen golden + inputs
WORK="${ST}/build/testcase/tadd"
mkdir -p "$WORK"
cp "${ST}/testcase/st_common.py" "$WORK/"
cp "${ST}/testcase/tadd/"{cases.py,gen_data.py,compare.py} "$WORK/"
cd "$WORK" && python3 gen_data.py   # ✅ verified

# 3) run main (blocked until build succeeds)
../../bin/tadd                      # ✅  runs CA model now

# 4) validate
python3 compare.py                  # ✅ verified
```
