First Triton Kernel: Vector Addition
=====================================
**Author:** `Alanna Burke <https://github.com/AlannaBurke>`_

.. grid:: 2

    .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn
       :class-card: card-prerequisites

       * How to shift from serial to parallel programming for GPUs
       * How to write a complete vector addition kernel in Triton
       * How to verify numerical correctness against PyTorch
       * How to benchmark and tune your kernel for performance

    .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites
       :class-card: card-prerequisites

       * PyTorch 2.3 or later
       * A GPU that supports Triton
       * Completion of the `Getting Started with Triton <https://pytorch.org/tutorials/intermediate/triton_getting_started.html>`_ tutorial

*This tutorial is derived from the* `SOTA Deep Learning Tutorials <https://www.youtube.com/@SOTADeepLearningTutorials>`_ *video series on Triton.*

Part 1: Making the Shift to Parallel Programming
-------------------------------------------------

In this tutorial, we will explore how to write a vector addition kernel using
Triton. First, we will cover the mental model shift required to move from
traditional serial programming to parallel programming. Understanding this
shift is crucial for effectively leveraging GPU architectures. Then, we will
dive into the actual coding of the Triton kernel.

Serial Approach
^^^^^^^^^^^^^^^

Consider a simple problem where we have two integer vectors, A and B, each
containing seven elements. Our goal is to compute their element-wise sum and
store the result in vector C.

In a conventional single-threaded programming model (e.g., Python or C++),
you would process one element at a time:

.. code-block:: python

    # Serial vector addition in Python
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [7, 6, 5, 4, 3, 2, 1]
    C = []
    for i in range(len(A)):
        C.append(A[i] + B[i])
    print(C)  # Output: [8, 8, 8, 8, 8, 8, 8]

Parallel Approach with Triton
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Triton introduces a different paradigm by leveraging parallelism on the GPU.
Instead of processing elements sequentially, we divide the workload across many
threads organized into blocks and grids. Each thread operates independently on
a subset of the data.

For example, if we choose a block size of 2, each block will have two threads.
For our problem of seven elements, we create a grid of four blocks (since
4 blocks x 2 threads = 8 threads, which covers all 7 elements plus one extra).
Each thread in a block processes one element of the vectors simultaneously.

Handling Edge Cases with Masking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A key challenge arises when the total number of elements is not a multiple of
the block size. In our example, the last block has two threads, but only one
element remains to be processed. The second thread in the last block has no
valid data to work on.

To handle this safely, we use **masking**. Each thread checks if its assigned
index is within the valid range of elements. Threads with indices outside the
valid range become inert and do not perform any load, compute, or store
operations. Proper masking prevents out-of-bounds memory access, which would
otherwise cause program crashes.

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def vector_add_kernel(
        A_ptr, B_ptr, C_ptr, N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)  # Each block has a unique program ID
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N  # Mask to avoid out-of-bounds
        a = tl.load(A_ptr + offsets, mask=mask)
        b = tl.load(B_ptr + offsets, mask=mask)
        tl.store(C_ptr + offsets, a + b, mask=mask)

    # The host program to launch the kernel
    def triton_vector_add(A, B):
        assert A.shape == B.shape
        N = A.numel()
        BLOCK_SIZE = 128
        C = torch.empty_like(A)
        grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)
        return C

    # Example usage
    A = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float32, device='cuda')
    B = torch.tensor([7, 6, 5, 4, 3, 2, 1], dtype=torch.float32, device='cuda')
    C = triton_vector_add(A, B)
    print(C.cpu().tolist())  # Output: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]

Key points in the code:

- ``tl.program_id(0)`` gets a unique ID in the grid.
- ``offsets`` computes the indices this block's threads will process.
- ``mask`` ensures threads do not access out-of-bounds memory.
- The kernel loads, adds, and stores elements in parallel.

Part 2: Writing the Kernel Step by Step
---------------------------------------

Now let's walk through the process of writing a vector addition kernel in more
detail, breaking down the host function and the kernel implementation.

Step 1: Importing Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by importing the necessary libraries:

.. code-block:: python

    import triton
    import triton.language as tl
    import torch

Step 2: Writing the Host Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The host function serves as the interface for users to run the kernel. It
takes two input vectors, A and B, and returns their element-wise sum as a
PyTorch tensor.

.. code-block:: python

    def vector_addition(A, B):
        output = torch.empty_like(A)
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.numel() == B.numel(), "Input tensors must have the same number of elements"
        num_elements = A.numel()
        block_size = 128
        def ceil_div(x, y):
            return (x + y - 1) // y
        grid_size = (ceil_div(num_elements, block_size),)
        kernel_vector[grid_size](A, B, output, num_elements, block_size)
        return output

Inside the host program:

- We create an output buffer with the same shape and device as A.
- We assert that both A and B are CUDA tensors and have the same size.
- We determine the number of elements to process.
- We define a block size to chunk the workload.
- We compute the grid size using a ceiling division function to ensure all
  elements are covered.
- We call the Triton kernel with the prepared parameters.

Step 3: Writing the Triton Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The kernel performs the core computation on the GPU:

.. code-block:: python

    @triton.jit
    def kernel_vector(a_ptr, b_ptr, out_ptr, num_elements, block_size: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * block_size
        thread_offsets = block_start + tl.arange(0, block_size)
        mask = thread_offsets < num_elements
        a_vals = tl.load(a_ptr + thread_offsets, mask=mask)
        b_vals = tl.load(b_ptr + thread_offsets, mask=mask)
        result = a_vals + b_vals
        tl.store(out_ptr + thread_offsets, result, mask=mask)

Key points:

- Retrieve the program ID to identify the current block.
- Calculate the starting offset for this block based on the block size and
  program ID.
- Compute thread offsets within the block.
- Apply a mask to prevent out-of-bounds memory access.
- Load elements from input vectors A and B using the computed offsets and mask.
- Perform element-wise addition.
- Store the result back to the output buffer with masking.

Marking the number of elements and block size as constant expressions allows
the compiler to optimize the kernel, which can lead to significant performance
improvements.

Step 4: Debugging Tips
^^^^^^^^^^^^^^^^^^^^^^

Triton provides a simple device print function for debugging purposes. While
it does not support formatted strings, it can be used to print values such as
the current program ID to the console during kernel execution.

.. code-block:: python

    # Inside the kernel (remove before production use):
    tl.device_print("pid: ", pid)

Part 3: Verifying Numerical Fidelity
-------------------------------------

After implementing a Triton kernel, the next critical step is to verify that
it produces correct results. We will use PyTorch's ``torch.allclose`` API to
compare outputs and ensure correctness.

Setting Up the Test Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    torch.manual_seed(0)  # Seed CPU and GPU RNGs for reproducibility

Creating Test Vectors
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    vector_size = 8192
    A = torch.rand(vector_size, device='cuda')
    B = torch.rand_like(A)

Computing the Reference Result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    torch_result = A + B

Running the Triton Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    triton_result = vector_addition(A, B)

Comparing Results
^^^^^^^^^^^^^^^^^

.. code-block:: python

    result_correct = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-4)
    print("Numerical fidelity correct:", result_correct)

Notes on tolerances:

- The absolute tolerance (``atol``) and relative tolerance (``rtol``) may need
  adjustment depending on the operation and data.
- For more complex operations, tighter tolerances may be needed to ensure
  correctness.

Wrapping the Verification in a Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def verify_numerics():
        torch.manual_seed(0)
        vector_size = 8192
        A = torch.rand(vector_size, device='cuda')
        B = torch.rand_like(A)
        torch_result = A + B
        triton_result = vector_addition(A, B)
        result_correct = torch.allclose(torch_result, triton_result, atol=1e-6, rtol=1e-4)
        print("Numerical fidelity correct:", result_correct)
        return result_correct

    if __name__ == "__main__":
        verify_numerics()

Troubleshooting Common Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Ensure that input tensors are on CUDA devices (``.is_cuda`` property).
- Use ``torch.manual_seed`` to seed both CPU and GPU RNGs for reproducibility.
- When calling ``.is_cuda``, use it as a property, not a method (i.e., no
  parentheses).
- If results are incorrect, check that your mask is applied consistently to
  both ``tl.load`` and ``tl.store``.

Part 4: Benchmarking and Performance Tuning
-------------------------------------------

With a correct kernel in hand, this part focuses on benchmarking and tuning
the kernel to optimize its performance.

Setting Up the Benchmarking Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To comprehensively evaluate performance, we use a benchmarking function that
runs the kernel over a range of vector sizes. The sizes are powers of two,
starting from 2\ :sup:`10` (1024) up to 2\ :sup:`28`, covering a broad
spectrum of tensor sizes.

Running the Benchmark
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time

    providers = [
        {"name": "PyTorch", "color": "blue"},
        {"name": "Triton", "color": "orange"}
    ]

    def benchmark_addition(vector_size, provider, block_size=128, num_warps=4):
        A = torch.randn(vector_size, device='cuda', dtype=torch.float32)
        B = torch.randn(vector_size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(A)
        start = time.time()
        if provider == "PyTorch":
            output = A + B
        elif provider == "Triton":
            kernel_vector[(vector_size // block_size,)](
                A, B, output, vector_size, block_size, num_warps=num_warps
            )
        torch.cuda.synchronize()
        end = time.time()
        gbps = (A.numel() * A.element_size() * 2) / (end - start) / 1e9
        return gbps

Interpreting Initial Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After running the initial benchmarks, you will observe that Triton's
performance largely matches PyTorch's across most tensor sizes. There may be a
slight performance gap in the mid-range sizes, but at larger tensor sizes,
Triton begins to match or exceed PyTorch.

Performance Tuning: Adjusting Block Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To improve performance, increase the block size from 128 to 1024 and rerun the
benchmark. A larger block size allows each kernel instance to process more
elements per launch, which can improve GPU utilization and reduce launch
overhead.

.. code-block:: python

    block_size = 1024
    # Rerun benchmark with new block size

Performance Tuning: Adjusting Number of Warps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, tune the number of warps used by the kernel. The default is 4 warps
(128 threads), but increasing to 8 warps allows the GPU to better hide memory
latency by switching between warps while one is waiting for data.

.. code-block:: python

    num_warps = 8
    kernel_vector[(vector_size // block_size,)](
        A, B, output, vector_size, block_size, num_warps=num_warps
    )

Final Benchmark Results
^^^^^^^^^^^^^^^^^^^^^^^

With the increased block size and number of warps, the benchmark shows that
Triton kernel performance matches or slightly exceeds PyTorch's performance
across the tested range.

Conclusion
----------

In this tutorial, you have:

1. Learned the mental model shift from serial to parallel programming.
2. Written a complete vector addition kernel in Triton with a host function.
3. Verified the numerical correctness of your kernel against PyTorch.
4. Benchmarked and tuned the kernel for optimal performance.

These skills form the foundation for writing more complex Triton kernels.

Further Reading
---------------

- `Writing a Triton Softmax Kernel <https://pytorch.org/tutorials/intermediate/triton_softmax_tutorial.html>`_
- `Using User-Defined Triton Kernels with torch.compile <https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>`_
- `Introduction to Triton <https://pytorch.org/docs/stable/user_guide/torch_compiler/triton/intro_to_triton.html>`_ (PyTorch Docs)
- `Triton language documentation <https://triton-lang.org/>`_
