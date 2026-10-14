Hf3fs
======================

An L2 adapter backed by the native C++ 3FS connector. Uses the 3FS Usrbio
API for high-performance distributed KV cache storage.

**Prerequisites**

- **3FS Usrbio**
  3FS Usrbio header (``hf3fs_usrbio.h``) and lib (``libhf3fs_api_shared.so``)
  are required when compiling the LMCache hf3fs extension.  We recommend
  installing them from source:

  .. code-block:: bash

      git clone https://github.com/deepseek-ai/3fs
      cd 3fs
      git submodule update --init --recursive
      ./patches/apply.sh
      Install dependencies
      cmake --build build -j 32
      cmake --install build

  See `3FS Build <https://github.com/deepseek-ai/3FS/blob/main/README.md#build-3fs>`_
  for the full build instructions.

  If 3FS Usrbio header or lib are not installed in the system path (e.g.,
  ``/usr/local/include``, ``/usr/local/lib``), you must set
  ``HF3FS_INCLUDE_DIR`` and ``HF3FS_LIB_DIR`` to point to them explicitly:

  .. code-block:: bash

      HF3FS_INCLUDE_DIR=/path/to/usrbio/include \
      HF3FS_LIB_DIR=/path/to/usrbio/lib \

- **libabsl-dev**

  The connector uses Abseil containers (``sharded_flat_hash_map``,
  ``flat_hash_map``) from the system library.  On Debian/Ubuntu:

  .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y libabsl-dev

**Build**

The LMCache 3FS extension is **not** built by default.  You must explicitly enable it:

.. code-block:: bash

    BUILD_HF3FS=1 \
    pip install -e . --verbose

The **BUILD_HF3FS** environment variable controls compilation:

- ``BUILD_HF3FS=1``: Enable build 3FS C++ extension.
- ``BUILD_HF3FS=0``: Force disable (highest priority), even if
  ``HF3FS_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``HF3FS_INCLUDE_DIR``. If
  ``HF3FS_INCLUDE_DIR`` is also unset, the 3FS extension is skipped.

**LMCache-Specific Fields**

- ``num_workers`` (int, default ``4``): C++ worker threads for I/O.
- ``per_op_workers`` (``dict[str, int]``, optional): A dict mapping lane keys
  to dedicated worker thread counts.  Supported keys:

  - ``"lookup"`` — threads for ``EXISTS`` operations.
  - ``"retrieve"`` — threads for ``GET``/load operations.
  - ``"store"`` — threads for ``SET``/put operations.
  - ``"delete"`` — threads for ``DELETE`` operations.

  Operations whose lane key is **not** present in the dict use the
  shared ``num_workers`` pool.  There is no requirement to set all
  keys — you can configure only the lanes that need dedicated pools.

**HF3FS Fields**

- ``mount_point`` (str, required): 3FS mount point directory.
- ``base_paths`` (str, required): Comma-separated subdirectories under
  ``mount_point``.
- ``ior_entries`` (int, default ``16``, range ``[1, 1024]``): Max I/Os a
  worker keeps in its ring at once (see *Batched I/O*).
- ``numa_id`` (int, default ``-1``): NUMA node ID (``-1`` = current node).
- ``single_buffer_size`` (int, default ``8388608`` (8 MiB), ``> 0``): Bytes
  of I/O buffer per ring entry, and the largest single I/O; larger keys are
  split into several I/Os.  Each worker allocates a read and a write buffer
  of ``single_buffer_size * ior_entries`` bytes each, so the defaults use
  256 MiB of shared memory per worker.
- ``enable_key_buffer`` (bool, default ``True``): Enable the in-memory
  key buffer for accelerated exists lookups.  When enabled, exist lookups
  respond from a hash set instead of issuing remote 3FS metadata calls.
  This dramatically reduces lookup latency.

**Configuration Example**

.. code-block:: bash

    --l2-adapter '{
      "type": "hf3fs",
      "mount_point": "/3fs/stage",
      "base_paths": "/3fs/stage/path1,/3fs/stage/path2",
      "ior_entries": 16,
      "numa_id": -1,
      "single_buffer_size": 8388608,
      "enable_key_buffer": true,
      "num_workers": 4,
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      }
    }'


**Batched I/O**

Each worker owns one read ring and one write ring (Usrbio ``Ior``), each
backed by its own ``Iov`` buffer.  The ``Iov`` is ``ior_entries`` fixed buffers
of ``single_buffer_size`` bytes, one per ring entry.  Instead of issuing one
I/O per key and waiting for it, a worker runs its share of a batch as a
bounded pipeline:

1. Open and register (``hf3fs_reg_fd``) the next key's file and prepare its
   I/O in a free ring entry, using that entry's buffer (keys larger than
   ``single_buffer_size`` are split into several I/Os).
2. Submit immediately (``hf3fs_submit_ios``) and keep preparing further keys
   while the ring holds fewer than ``ior_entries`` I/Os.  When it is full,
   wait for one completion and reuse its entry and buffer.
3. Reap completions as they arrive.  A read's data is copied out of the
   ``Iov`` as soon as its I/O lands, so the copy overlaps the I/Os still in
   flight instead of waiting for the whole batch.
4. Deregister and close each file once its last I/O has completed.

Every ring is created with 3FS ``io_depth`` ``0``, so the 3FS ring worker
starts each I/O as soon as it sees it.  This is not configurable: a non-zero
``io_depth`` holds back batches smaller than ``|io_depth|``, and 3FS re-checks a
held-back batch only when a later submit wakes its ring watcher, so the last
request before the client goes idle would never complete.

For example, with the defaults and 8 MiB KV chunks, a worker keeps 16 chunks
in flight and starts the next one as soon as any of them completes, where the
earlier design waited on each chunk before starting the next.  Because each
entry owns its buffer, a completed I/O frees its buffer immediately, even
when it finishes before earlier ones.

A failed key does not stop the rest of the batch: a load reports only that
key as missing, and a store still writes every other key but reports the
batch as failed.

**In-Memory Key Buffer**

When ``enable_key_buffer`` is ``True`` (the default), the connector maintains an
in-memory cache of all key filenames. Lookup operations respond from the cache instead
of issuing remote 3FS metadata calls. This dramatically reduces lookup latency.

- **Trade-offs**
   - **Enable** for workloads with many concurrent lookups (prefetch,batch exists). 
  The latency savings outweigh the memory cost.
   - **Disable**  when memory is constrained.

To disable the buffer cache:

.. code-block:: bash

    --l2-adapter '{
      "type": "hf3fs",
      "mount_point": "/3fs/stage",
      "base_paths": "/3fs/stage/path1",
      "ior_entries": 16,
      "numa_id": -1,
      "single_buffer_size": 8388608,
      "enable_key_buffer": false,
    }'


**Unit Tests**

The hf3fs tests are gated on 3FS extension availability and 3FS storage
cluster accessibility:
- 3FS C++ extension missing → config tests only
- 3FS C++ extension present and 3FS storage cluster available → full test suite

To run the full test suite, set the **HF3FS_MOUNT_POINT** environment
variable to the mount point directory of a real 3FS storage cluster:

.. code-block:: bash

    HF3FS_MOUNT_POINT=/3fs/stage \
    pytest tests/v1/distributed/test_hf3fs_l2_adapter.py -v