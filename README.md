# go-faiss

Go bindings for [Faiss](https://github.com/facebookresearch/faiss).

## Install

First you will need to build and install Faiss. The `C-api-cmake` branch at
https://github.com/DataIntelligenceCrew/faiss/tree/C-api-cmake has some fixes
for building the C API.

```sh
git clone https://github.com/DataIntelligenceCrew/faiss.git
cd faiss
git checkout C-api-cmake
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON
make -C build
sudo make -C build install
```

Building will produce the dynamic library `faiss_c`.
You will need to install it in a place where your system will find it (e.g.
`/usr/lib` on Linux).
You can do this with:

    sudo cp build/c_api/libfaiss_c.so /usr/lib

Now you can install the Go module:

    go get github.com/DataIntelligenceCrew/go-faiss
