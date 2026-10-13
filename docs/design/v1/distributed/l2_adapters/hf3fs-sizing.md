# Native HF3FS ring and buffer sizing

The native HF3FS adapter budgets bytes per ring entry so increasing ring
capacity also increases the data buffer available to that ring. This matches
the sizing model of the vLLM and SGLang 3FS clients.

## Configuration contract

- `ior_entries` defaults to 8 and must be a positive integer. There is no
  adapter-specific minimum of 128 or maximum of 1024.
- `single_buffer_size` defaults to 16 MiB and must be a positive integer.
- Each worker owns read and write rings, each with a separate Iov of
  `single_buffer_size * ior_entries` bytes. Both directions are created when
  the worker starts, including workers dedicated to one operation lane.
- The Python config exposes the derived total through its read-only
  `iov_size` property. The factory passes the per-entry size to C++, which
  computes the allocation size and rejects multiplication overflow before
  creating any 3FS resources.

For example, `ior_entries=8` and `single_buffer_size=16777216` allocate
128 MiB per direction, or 256 MiB per worker plus ring metadata.

## Batch execution

The native connector retains its existing worker-tile scheduling:

```text
request -> roughly equal worker tiles
        -> pack each tile into waves bounded by ring entries and Iov bytes
        -> submit once per wave -> reap all completions -> reuse the Iov
```

`single_buffer_size` controls allocation, not a maximum key size. A key larger
than the total Iov is segmented across waves. With uniform chunks equal to
`single_buffer_size`, a wave holds at most `ior_entries` chunks.

The former whole-buffer `iov_size` option is rejected to prevent old
configurations from silently using a different allocation. To preserve an
old total, use `single_buffer_size = ceil(old_iov_size / ior_entries)`.
The separate legacy Python HF3FS connector keeps its own configuration.
