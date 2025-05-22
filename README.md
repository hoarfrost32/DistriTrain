## For server -
```bash
mpic++ -std=c++17 server.cpp -I/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/include   -I/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/include/torch/csrc/api/include   -L/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/lib   -ltorch -lc10   -Wl,-rpath,/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/lib -ltorch_cpu -pthread -D_GLIBCXX_USE_CXX11_ABI=0
```
## For client - 
```bash
mpic++ -std=c++17 client.cpp -I/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/include   -I/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/include/torch/csrc/api/include   -L/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/lib   -ltorch -lc10   -Wl,-rpath,/home2/keshav06/miniconda3/envs/keshav/lib/python3.8/site-packages/torch/lib -ltorch_cpu  -o mn   -pthread -D_GLIBCXX_USE_CXX11_ABI=0
```
## For running - 
```bash
mpirun --hostfile hosts.txt -np 1 ./a.out : -np 4 ./mn
```
### For simulating failure of rank 1 process, make the global variable in server.cpp to true