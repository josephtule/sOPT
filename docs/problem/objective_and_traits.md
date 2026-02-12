# Objective Shape and Traits

## Objective shape

The minimum objective API is:

```cpp
struct Obj {
    f64 func(ecref<vecXd> x) const;
};
```

Optional methods:

```cpp
void gradient(ecref<vecXd> x, eref<vecXd> g) const;
void hessian(ecref<vecXd> x, eref<matXd> H) const;
void hessian_vector(ecref<vecXd> x, ecref<vecXd> v, eref<vecXd> Hv) const;
```

## Traits (`include/sOPT/problem/traits.hpp`)

- `has_gradient_v<T>`
- `has_hessian_v<T>`
- `has_hessian_vector_v<T>`

These traits are used by `Oracle<T>` to choose analytic derivative paths when available and finite-difference fallbacks otherwise.
