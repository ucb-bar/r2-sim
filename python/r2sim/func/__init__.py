"""The functional plane in Python: the backend that the plug-in
libr2_python.so runs inside the r2-sim process (`--functional python`).

`memory` holds the bytes (arrays over r2-sim's store); `descriptor` decodes
the descriptor bytes of a task; `ops` has one function per task kind (dma,
matmul, vector); `plane` is the state (memory and the Atlas accumulators);
`backend` is the module that the plug-in calls."""
