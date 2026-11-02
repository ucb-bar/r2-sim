"""The functional plane in Python: the backend that the plug-in
libr2_python.so runs inside the r2-sim process (`--functional python`).

`memory` holds the bytes (arrays over r2-sim's store); `descriptor` decodes
the descriptor bytes of a task; `ops` has one function per task kind (dma,
matmul, vector) and `kernels` the Muon kernels of the registry, both shared
by the providers; `provider_numpy` and `provider_torch` hold the memory
access and the arithmetic of each provider; `plane` is the state (memory,
the provider, the Atlas accumulators and the kernels); `backend` is the
module that the plug-in calls."""
