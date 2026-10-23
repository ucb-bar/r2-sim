"""The functional plane in Python: r2sim runs it as a child process
(`--functional python`) and forwards every op to it (protocol: `protocol`).

`memory` holds the bytes; `ops` has one function per op kind (dma, matmul,
vector, store); `plane` is the state (memory and the Atlas accumulators)."""
