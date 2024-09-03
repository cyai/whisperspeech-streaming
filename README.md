### Commands to run in server:
```bash
cd
apt-get update
apt-get install -y lsof unzip llvm uvicorn  build-essential python3-dev
git clone https://github.com/cyai/whisperspeech-streaming
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
apt-get update
wget https://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb
apt install ./libffi6_3.2.1-8_amd64.deb
apt-get install -y llvm-9
cd whisperspeech-streaming
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc
pip install poetry
poetry install --no-root --no-interaction --no-ansi

pip install numba llvmlite

```

https://askubuntu.com/questions/1286131/how-do-i-install-llvm-10-on-ubuntu-18-04

<!-- apt-get install build-essential curl libcap-dev git cmake libncurses5-dev python-pip unzip libtcmalloc-minimal4 libgoogle-perftools-dev libsqlite3-dev doxygen python3 gcc-multilib g++-multilib 
pip install lit tabulate wllvm -->

wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
apt-get update
apt-get install -y llvm-9