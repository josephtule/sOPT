#pragma once

#include "sOPT/core/vecdefs.hpp"
#include <type_traits>

namespace sOPT {

// checks if type has a gradient attached in the correct form ------------------
template <typename T, typename = void>
struct has_gradient : std::false_type {};

template <typename T>
struct has_gradient<
    T,
    std::void_t<decltype(std::declval<const T&>().gradient(
        std::declval<ecref<vecXd>>(),
        std::declval<eref<vecXd>>()
    ))>> : std::true_type {};

template <typename T>
inline constexpr bool has_gradient_v = has_gradient<T>::value;

// checks if type has a hessian attached in the correct form -------------------
template <typename T, typename = void>
struct has_hessian : std::false_type {};

template <typename T>
struct has_hessian<T, std::void_t<decltype(std::declval<const T&>().hessian(
    std::declval<ecref<vecXd>>(),
    std::declval<eref<matXd>>()
))>> : std::true_type {};

template <typename T>
inline constexpr bool has_hessian_v = has_hessian<T>::value;

// checks if type has a hessian-vector product in the correct form --------------
template <typename T, typename = void>
struct has_hessian_vector : std::false_type {};

template <typename T>
struct has_hessian_vector<
    T,
    std::void_t<decltype(std::declval<const T&>().hessian_vector(
        std::declval<ecref<vecXd>>(),
        std::declval<ecref<vecXd>>(),
        std::declval<eref<vecXd>>()
    ))>> : std::true_type {};

template <typename T>
inline constexpr bool has_hessian_vector_v = has_hessian_vector<T>::value;

// checks if type has a residual function in the correct form ------------------
template <typename T, typename = void>
struct has_residual : std::false_type {};

template <typename T>
struct has_residual<
    T,
    std::void_t<decltype(std::declval<const T&>().residual(
        std::declval<ecref<vecXd>>(),
        std::declval<eref<vecXd>>()
    ))>> : std::true_type {};

template <typename T>
inline constexpr bool has_residual_v = has_residual<T>::value;

// checks if type has a jacobian function in the correct form ------------------
template <typename T, typename = void>
struct has_jacobian : std::false_type {};

template <typename T>
struct has_jacobian<
    T,
    std::void_t<decltype(std::declval<const T&>().jacobian(
        std::declval<ecref<vecXd>>(),
        std::declval<eref<matXd>>()
    ))>> : std::true_type {};

template <typename T>
inline constexpr bool has_jacobian_v = has_jacobian<T>::value;

} // namespace sOPT
