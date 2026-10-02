set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
mkdir -p "$TASK_DIR"/src "$TASK_DIR"/build "$TASK_DIR"/logs "$TASK_DIR"/cache "$TASK_DIR"/tmp
cd "$TASK_DIR"
export XDG_CACHE_HOME="$TASK_DIR/cache" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache"
for source in pyre altar; do
  if [ ! -d "src/$source/.git" ]; then git clone "https://github.com/lijun99/$source.git" "src/$source" > "logs/clone-$source.txt" 2>&1; fi
  if [ "$(git -C "src/$source" rev-parse --is-shallow-repository)" = true ]; then
    git -C "src/$source" fetch --unshallow --tags > "logs/tags-$source.txt" 2>&1
  else
    git -C "src/$source" fetch --tags > "logs/tags-$source.txt" 2>&1
  fi
 done
git -C src/pyre checkout --detach 86ff6185625139200c9e4d2f8a655c0911fd4645
git -C src/altar checkout --detach 6646198a928d0ea3c24f8d32b4e04b2fc4fa471a
cmake -S src/pyre -B build/pyre-gcc11 -DCMAKE_INSTALL_PREFIX="$TASK_DIR/native" -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS > logs/configure-pyre-gcc11.txt 2>&1
cmake --build build/pyre-gcc11 -j 8 > logs/build-pyre.txt 2>&1
cmake --install build/pyre-gcc11 > logs/install-pyre.txt 2>&1
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/native/packages"
cmake -S src/altar -B build/altar -DCMAKE_INSTALL_PREFIX="$TASK_DIR/native" -DCMAKE_PREFIX_PATH="$TASK_DIR/native" -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release > logs/configure-altar.txt 2>&1
cmake --build build/altar -j 8 > logs/build-altar.txt 2>&1
cmake --install build/altar > logs/install-altar.txt 2>&1
python3 -c 'import pyre,altar,cuda; print(pyre.__file__,altar.__file__,cuda.__file__)' > logs/native-imports.txt 2>&1
cat logs/native-imports.txt
