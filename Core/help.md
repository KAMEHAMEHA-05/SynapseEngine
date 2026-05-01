# Core Module Reference

---

## tensor.jl

### `abstract type BackwardOp end`

Marker type for all backward operation structs.

---

### `mutable struct Tensor{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}`

**Fields:**

* `data::A`
  Underlying array storing values.

* `grad::Union{Nothing, A}`
  Gradient buffer (same shape as `data`). `nothing` if not initialized.

* `parents::Vector{Tensor}`
  Tensors that produced this tensor in the computation graph.

* `backward::Union{Nothing, BackwardOp}`
  Backward operation associated with this tensor.

---

### `Tensor(data::AbstractArray{T,N})`

**Parameters:**

* `data`: input array

**Returns:**

* New `Tensor` with:

  * `grad = nothing`
  * `parents = []`
  * `backward = nothing`

---

### `Base.size(t::Tensor)`

**Parameters:**

* `t`: Tensor

**Returns:**

* Size of `t.data`

---

### `Base.getindex(t::Tensor, I...)`

**Parameters:**

* `t`: Tensor
* `I...`: indices

**Returns:**

* Indexed value from `t.data`

---

### `Base.setindex!(t::Tensor, v, I...)`

**Parameters:**

* `t`: Tensor
* `v`: value
* `I...`: indices

**Effect:**

* Sets `t.data[I...] = v`

---

### `Base.IndexStyle(::Type{<:Tensor})`

**Returns:**

* Uses `Array` indexing style

---

### `ensure_grad!(t::Tensor)`

**Parameters:**

* `t`: Tensor

**Effect:**

* If `t.grad === nothing`:

  * Allocates gradient buffer using `similar(t.data)`
  * Fills with zeros

---

## ops.jl

### `const OpCacheKey = Tuple{UInt, UInt, DataType}`

Key format used for operation caching:

* `UInt`: objectid of first tensor
* `UInt`: objectid of second tensor
* `DataType`: backward op type

---

### Binary Ops: `+`, `-`, `*`, `/`

#### Signature

```julia
(a::Tensor, b::Tensor, cache::Union{Nothing, Dict{OpCacheKey, Tensor}}=nothing)
```

**Parameters:**

* `a`, `b`: input tensors
* `cache`: optional cache dictionary

**Behavior:**

* If cache hit:

  * Reuse tensor
  * Update `.data`
* Else:

  * Create new tensor
  * Set:

    * `parents = [a, b]`
    * `backward = <OpBackward>(a, b)`

**Returns:**

* Output tensor

---

### `backward!(op::<OpBackward>, grad)`

**Parameters:**

* `op`: backward op instance
* `grad`: upstream gradient

**Effect:**

* Accumulates gradients into parent tensors using `.+=`

---

### `matmul(a::Tensor, b::Tensor, cache=nothing)`

**Parameters:**

* `a`, `b`: tensors
* `cache`: optional cache

**Behavior:**

* Uses:

  * `*` for 2D
  * `NNlib.batched_mul` for higher dims
* Same caching pattern as other ops

**Returns:**

* Output tensor

---

### `sum(t::Tensor, cache=nothing)`

**Parameters:**

* `t`: tensor
* `cache`: optional cache

**Returns:**

* Tensor with scalar sum (wrapped as 1-element array)

---

### `reduce_sum(t::Tensor, dims; keepdims=false, cache=nothing)`

**Parameters:**

* `t`: tensor
* `dims`: dimensions to reduce
* `keepdims`: whether to retain reduced dimensions
* `cache`: optional cache

**Returns:**

* Reduced tensor

---

### Activation Functions

#### `ReLU(t::Tensor; cache=nothing)`

Applies elementwise `max(x, 0)`

#### `LeakyReLU(t::Tensor, alpha=0.01f0; cache=nothing)`

Applies:

* `x` if `x > 0`
* `alpha * x` otherwise

#### `Linear(t::Tensor; cache=nothing)`

Identity operation (copy of input)

#### `softmax(t::Tensor, dims=1; cache=nothing)`

Applies softmax along given dimension

---

### Backward Ops

Each op defines a struct:

```julia
struct <OpName>Backward <: BackwardOp
    ...
end
```

And corresponding:

```julia
backward!(op::<OpName>Backward, grad)
```

**Effect:**

* Computes local gradients
* Accumulates into parent `.grad`

---

## autograd.jl

### `clip_grad_tensor!(p::Tensor, max_norm::Real)`

**Parameters:**

* `p`: tensor
* `max_norm`: maximum allowed L2 norm

**Effect:**

* Scales `p.grad` if its norm exceeds `max_norm`

---

### `quick_sig(loss::Tensor, param_ids::Set{UInt})`

**Parameters:**

* `loss`: output tensor
* `param_ids`: set of parameter object IDs

**Returns:**

* Hash based on immediate and second-level parents

---

### `build_topo(t::Tensor)`

**Parameters:**

* `t`: tensor

**Returns:**

* Vector of tensors in topological order

**Behavior:**

* DFS traversal over `parents`

---

### `graph_signature(topo::Vector{Tensor}, param_ids::Set{UInt})`

**Parameters:**

* `topo`: topologically sorted tensor list
* `param_ids`: parameter IDs

**Returns:**

* Hash over nodes present in `param_ids`

---

### `backprop(loss::Tensor)`

**Parameters:**

* `loss`: final scalar tensor

**Behavior:**

1. Build topological ordering
2. Reset gradients for all nodes
3. Initialize `loss.grad = 1`
4. Traverse nodes in reverse order
5. For each node:

   * Ensure gradient exists
   * Call `backward!` if defined

**Effect:**

* Populates `.grad` for all tensors in graph

---
