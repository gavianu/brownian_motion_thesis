import os, numpy as np
print("OMP_NUM_THREADS=", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS=", os.environ.get("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS=", os.environ.get("OPENBLAS_NUM_THREADS"))

# vezi cu ce e legat NumPy
np.__config__.show()

# (opțional, foarte util)
# pip install threadpoolctl
from threadpoolctl import threadpool_info
for lib in threadpool_info():
    print(lib["internal_api"], lib.get("filename"), lib.get("num_threads"))
