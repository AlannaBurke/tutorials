Writing a Triton Softmax Kernel
================================
**Author:** `Alanna Burke <https://github.com/AlannaBurke>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to implement naive and online softmax in PyTorch
       * How to write a custom softmax kernel in Triton
       * How to verify numerical correctness against PyTorch
       * How to benchmark your Triton softmax against PyTorch and a naive implementation

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.3 or later
       * A GPU that supports Triton
       * Completion of the `First Triton Kernel: Vector Addition <https://pytorch.org/tutorials/intermediate/triton_first_kernel_tutorial.html>`_ tutorial

*This tutorial is derived from the* `SOTA Deep Learning Tutorials <https://www.youtube.com/@SOTADeepLearningTutorials>`_ *video series on Triton.*

Part 1: Basic Softmax and Online Softmax in PyTorch
----------------------------------------------------

Before writing a Triton kernel, it is important to understand the operation
we are implementing. In this part, we implement both a naive and an online
softmax in PyTorch and verify their correctness.

Environment Setup
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install torch triton

Creating a Sample Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch.nn.functional as F

    sample = torch.tensor(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        dtype=torch.float32,
        device='cuda'
    )

Reference: PyTorch Softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    softmax_ref = F.softmax(sample, dim=1)
    print("PyTorch Softmax Output:\n", softmax_ref)

Naive Softmax: Implementation and Inefficiencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's implement our own numerically stable softmax function in PyTorch:

.. code-block:: python

    def naive_softmax(x: torch.Tensor) -> torch.Tensor:
        # Step 1: Subtract the row-wise max for numerical stability
        x_max = x.max(dim=1, keepdim=True).values
        x_safe = x - x_max
        # Step 2: Exponentiate
        numerator = torch.exp(x_safe)
        # Step 3: Sum across rows for denominator
        denominator = numerator.sum(dim=1, keepdim=True)
        # Step 4: Divide to get softmax
        return numerator / denominator

    softmax_naive = naive_softmax(sample)
    assert torch.allclose(softmax_naive, softmax_ref, atol=1e-6), "Mismatch!"

This implementation requires multiple passes over each row of data:

1. Find the maximum value (for numerical stability) -- 1st pass.
2. Subtract the max and exponentiate -- 2nd pass.
3. Sum the exponentiated values -- 3rd pass.
4. Divide to get the final softmax -- 4th pass.

Each pass reads the entire row from HBM, which is costly for large tensors.

Online Softmax: Reducing Memory Passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Online softmax reduces the number of passes to just two by maintaining running
statistics -- the current row maximum and the running sum of exponentials -- as
it processes each element. When the running maximum is updated, the accumulated
sum is rescaled to remain consistent.

.. code-block:: python

    def online_softmax(x: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = x.shape
        output = torch.zeros_like(x)
        for r in range(n_rows):
            row_max = float('-inf')
            normalizer = 0.0
            # First pass: compute row_max and normalizer together
            for c in range(n_cols):
                val = x[r, c].item()
                prev_row_max = row_max
                row_max = max(row_max, val)
                if row_max != prev_row_max:
                    # Rescale the running sum for the new maximum
                    normalizer = normalizer * torch.exp(
                        torch.tensor(prev_row_max - row_max)
                    )
                normalizer += torch.exp(torch.tensor(val - row_max))
            # Second pass: compute final softmax values
            for c in range(n_cols):
                output[r, c] = torch.exp(x[r, c] - row_max) / normalizer
        return output

    # Verify correctness
    softmax_online = online_softmax(sample.cpu()).to(sample.device)
    assert torch.allclose(softmax_online, softmax_ref.cpu(), atol=1e-6), "Mismatch!"
    print("Online softmax matches PyTorch softmax.")

In practice, the online softmax is significantly faster than the naive version
for large tensors, because it reduces the number of passes over each row from
four to two.

Part 2: Softmax in Triton
--------------------------

Now we implement a custom softmax kernel in Triton, including both the kernel
itself and the host program that launches it. We will cover memory management,
parallelization, pointer arithmetic, and benchmark our implementation.

Understanding Triton Kernels and Host Programs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When working with Triton, you typically write two components:

- **The Kernel:** The function that runs on the GPU, processing data in
  parallel.
- **The Host Program:** The Python code that sets up meta-information (like
  block size and grid shape), allocates memory, and launches the kernel.

The Softmax Kernel and Host Program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import triton
    import triton.language as tl

    def next_power_of_2(x):
        return 1 << (x - 1).bit_length()

    @triton.jit
    def softmax_kernel(
        output_ptr, output_stride,
        input_ptr, input_stride,
        n_cols, BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        input_row_ptr = input_ptr + row_idx * input_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(input_row_ptr + col_offsets, mask=mask, other=float('-inf'))
        row_max = tl.max(row, axis=0)
        row = row - row_max
        numerator = tl.exp(row)
        denom = tl.sum(numerator, axis=0)
        softmax = numerator / denom
        output_row_ptr = output_ptr + row_idx * output_stride
        tl.store(output_row_ptr + col_offsets, softmax, mask=mask)

    def triton_softmax(x):
        assert x.ndim == 2, "Only 2D tensors are supported in this example."
        n_rows, n_cols = x.shape
        BLOCK_SIZE = next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE > 2048:
            num_warps = 8
        if BLOCK_SIZE > 4096:
            num_warps = 16
        grid = (n_rows,)
        output = torch.empty_like(x)
        softmax_kernel[grid](
            output, output.stride(0),
            x, x.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return output

Key Implementation Details
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Memory and Pointer Arithmetic:**
Tensors in memory are 1D arrays; strides are used to move between rows. The
kernel receives pointers to the input and output data, and uses pointer
arithmetic to access the correct row and column. Masking ensures we only
read/write valid data when the number of columns is not a power of two.

**Parallelization:**
We launch one kernel instance per row (``grid = (n_rows,)``). Each kernel
instance processes a full row, with the ``BLOCK_SIZE`` set to the next power
of two greater than or equal to the number of columns.

**Softmax Computation:**
For numerical stability, we subtract the row max before exponentiating. We sum
the exponentiated values to get the denominator, and write the result back to
the output buffer using the same mask.

Debugging and Common Pitfalls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pointer arithmetic is error-prone. A common mistake is to incorrectly
calculate the output pointer, leading to overwriting the same row multiple
times. Always ensure that:

- The output pointer is offset by both the row index (via ``output_stride``)
  and the column offset.
- Masking is applied consistently when both reading and writing.

If you see duplicated or inverted rows in your output, double-check your
pointer calculations.

Numerical Verification
^^^^^^^^^^^^^^^^^^^^^^

Before benchmarking, always verify correctness:

.. code-block:: python

    import torch.nn.functional as F

    x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
    y_triton = triton_softmax(x)
    y_torch = F.softmax(x, dim=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-6), "Numerical mismatch!"
    print("Triton softmax matches PyTorch softmax.")

Benchmarking
^^^^^^^^^^^^

Then benchmark across a range of tensor sizes:

.. code-block:: python

    import time

    def naive_softmax(x):
        x_max = x.max(dim=1, keepdim=True).values
        x_safe = x - x_max
        numerator = torch.exp(x_safe)
        denominator = numerator.sum(dim=1, keepdim=True)
        return numerator / denominator

    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    for n_cols in sizes:
        x = torch.randn(1024, n_cols, device='cuda', dtype=torch.float32)
        start = time.time(); y_triton = triton_softmax(x); triton_time = time.time() - start
        start = time.time(); y_torch = F.softmax(x, dim=1); torch_time = time.time() - start
        start = time.time(); y_naive = naive_softmax(x); naive_time = time.time() - start
        print(
            f"Cols: {n_cols} | Triton: {triton_time:.4f}s "
            f"| PyTorch: {torch_time:.4f}s | Naive: {naive_time:.4f}s"
        )

Typically, you will see:

- **Triton:** Outperforms naive and matches or exceeds PyTorch for large
  tensors.
- **PyTorch:** Fast for small tensors, but may decline as size increases.
- **Naive:** Significantly slower than both.

Conclusion
----------

In this tutorial, you have:

1. Implemented naive and online softmax in PyTorch and verified their
   correctness.
2. Written a complete softmax kernel in Triton with pointer arithmetic and
   masking.
3. Verified the numerical correctness of your Triton kernel against PyTorch.
4. Benchmarked your Triton softmax against PyTorch and a naive implementation.

Further Reading
---------------

- `Using User-Defined Triton Kernels with torch.compile <https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>`_
- `Introduction to Triton <https://pytorch.org/docs/stable/user_guide/torch_compiler/triton/intro_to_triton.html>`_ (PyTorch Docs)
- `Triton Resources <https://pytorch.org/docs/stable/user_guide/torch_compiler/triton/resources.html>`_ (PyTorch Docs)
- `Triton language documentation <https://triton-lang.org/>`_
