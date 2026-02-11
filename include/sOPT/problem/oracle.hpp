#pragma once

#include "sOPT/core/options.hpp"
#include "sOPT/core/typedefs.hpp"
#include "sOPT/core/util.hpp"
#include "sOPT/core/vecdefs.hpp"
#include "sOPT/finite_diff/fd_grad.hpp"
#include "sOPT/problem/traits.hpp"
#include <cmath>

namespace sOPT {

template <typename Obj>
class Oracle {
  public:
    Oracle(const Obj& obj, const Options& opt)
        : obj_(obj), opt_(opt), f_cache_(opt.cache.enabled, opt.cache.f_slots),
          g_cache_(opt.cache.enabled, opt.cache.g_slots),
          h_cache_(opt.cache.enabled, opt.cache.h_slots) {}

    // try evals
    bool try_func(ecref<vecXd> x, f64& fx) {
        if (cache_lookup_(f_cache_, x, fx)) return true;
        if (!can_eval_f_()) return false;
        ++f_evals_;
        fx = obj_.func(x);
        if (!isfinite(fx)) return false;
        cache_store_(f_cache_, x, fx);
        return true;
    }
    bool try_gradient(ecref<vecXd> x, eref<vecXd> g) {
        if (cache_lookup_(g_cache_, x, g)) return true;
        if (!can_eval_g_()) return false;
        ++g_evals_;
        if constexpr (has_gradient_v<Obj>) {
            obj_.gradient(x, g);
        } else { // finite difference fallbacks
            switch (opt_.fd.fallback_grad) {
            case FallbackGrad::fd_forward:
                if (!fd_gradient_forward(*this, x, g, opt_.fd.eps)) return false;
                break;
            case FallbackGrad::fd_backward:
                if (!fd_gradient_backward(*this, x, g, opt_.fd.eps)) return false;
                break;
            case FallbackGrad::fd_central:
                if (!fd_gradient_central(*this, x, g, opt_.fd.eps)) return false;
                break;
            case FallbackGrad::fd_forward_2:
                if (!fd_gradient_forward_2(*this, x, g, opt_.fd.eps)) return false;
                break;
            case FallbackGrad::fd_backward_2:
                if (!fd_gradient_backward_2(*this, x, g, opt_.fd.eps)) return false;
                break;
            case FallbackGrad::fd_central_2:
                if (!fd_gradient_central_2(*this, x, g, opt_.fd.eps)) return false;
                break;
            }
        }
        if (!g.allFinite()) return false;
        cache_store_(g_cache_, x, g);
        return true;
    }
    bool try_hessian(ecref<vecXd> x, eref<matXd> H) {
        maybe_apply_hessian_guard_(static_cast<i32>(x.size()));
        if (cache_lookup_(h_cache_, x, H)) return true;
        if (!can_eval_h_()) return false;
        ++h_evals_;
        if constexpr (has_hessian_v<Obj>) {
            obj_.hessian(x, H);
        } else { // finite difference fallbacks
            switch (opt_.fd.fallback_hess) {
            case FallbackHess::fd_forward:
                if (!fd_hessian_forward(*this, x, H, opt_.fd.eps)) return false;
                break;
            case FallbackHess::fd_backward:
                if (!fd_hessian_backward(*this, x, H, opt_.fd.eps)) return false;
                break;
            case FallbackHess::fd_central:
                if (!fd_hessian_central(*this, x, H, opt_.fd.eps)) return false;
                break;
            case FallbackHess::fd_forward_2:
                if (!fd_hessian_forward_2(*this, x, H, opt_.fd.eps)) return false;
                break;
            case FallbackHess::fd_backward_2:
                if (!fd_hessian_backward_2(*this, x, H, opt_.fd.eps)) return false;
                break;
            case FallbackHess::fd_central_2:
                if (!fd_hessian_central_2(*this, x, H, opt_.fd.eps)) return false;
                break;
            }
        }
        if (!H.allFinite()) return false;
        cache_store_(h_cache_, x, H);
        return true;
    }
    bool try_hv(ecref<vecXd> x, ecref<vecXd> v, eref<vecXd> Hv) {
        const i32 n = static_cast<i32>(x.size());
        if (v.size() != n || Hv.size() != n) return false;

        ++hv_evals_;
        if constexpr (has_hessian_vector_v<Obj>) {
            obj_.hessian_vector(x, v, Hv);
            return Hv.allFinite();
        } else if constexpr (has_hessian_v<Obj>) {
            if (hv_H_.rows() != n || hv_H_.cols() != n) hv_H_.resize(n, n);
            if (!try_hessian(x, hv_H_)) return false;
            Hv.noalias() = hv_H_.selfadjointView<eig::Lower>() * v;
            return Hv.allFinite();
        } else {
            switch (opt_.fd.fallback_hv) {
            case FallbackHv::fd_forward:
                return fd_hv_forward(*this, x, v, Hv, opt_.fd.hv_eps);
            case FallbackHv::fd_backward:
                return fd_hv_backward(*this, x, v, Hv, opt_.fd.hv_eps);
            case FallbackHv::fd_central:
                return fd_hv_central(*this, x, v, Hv, opt_.fd.hv_eps);
            case FallbackHv::fd_forward_2:
                return fd_hv_forward_2(*this, x, v, Hv, opt_.fd.hv_eps);
            case FallbackHv::fd_backward_2:
                return fd_hv_backward_2(*this, x, v, Hv, opt_.fd.hv_eps);
            case FallbackHv::fd_central_2:
                return fd_hv_central_2(*this, x, v, Hv, opt_.fd.hv_eps);
            }
        }
        return false;
    }

    // eval helpers
    i32 f_evals() const { return f_evals_; }
    i32 g_evals() const { return g_evals_; }
    i32 h_evals() const { return h_evals_; }
    i32 hv_evals() const { return hv_evals_; }

    // cache helpers
    i32 f_cache_slots() const { return f_cache_.slots(); }
    i32 g_cache_slots() const { return g_cache_.slots(); }
    i32 h_cache_slots() const { return h_cache_.slots(); }
    u64 f_cache_hits() const { return f_cache_.hits; }
    u64 f_cache_misses() const { return f_cache_.misses; }
    u64 g_cache_hits() const { return g_cache_.hits; }
    u64 g_cache_misses() const { return g_cache_.misses; }
    u64 h_cache_hits() const { return h_cache_.hits; }
    u64 h_cache_misses() const { return h_cache_.misses; }

    // limits
    static inline bool limit_enabled_(i32 v) { return v >= 0; }
    bool f_limit_reached() const {
        return limit_enabled_(opt_.limits.max_f_evals)
               && (f_evals_ >= opt_.limits.max_f_evals);
    }
    bool g_limit_reached() const {
        return limit_enabled_(opt_.limits.max_g_evals)
               && (g_evals_ >= opt_.limits.max_g_evals);
    }
    bool h_limit_reached() const {
        return limit_enabled_(opt_.limits.max_h_evals)
               && (h_evals_ >= opt_.limits.max_h_evals);
    }
    bool any_limit_reached() const {
        return f_limit_reached() || g_limit_reached() || h_limit_reached();
    }

  private:
    template <typename T>
    struct CacheEntry {
        bool has_value = false;
        vecXd x;          // cache entry key
        T value;          // cache entry value
        u64 lru_tick = 0; // least-recently-used tick
    };
    template <typename T>
    struct CacheSet {
        bool enabled = false;
        u64 tick = 0;
        u64 hits = 0;
        u64 misses = 0;
        svec<CacheEntry<T>> entries;

        CacheSet() = default;
        CacheSet(bool enabled_in, i32 slots) {
            const bool use_cache = enabled_in && (slots > 0);
            enabled = use_cache;
            if (use_cache) entries.resize(static_cast<i32>(slots));
        }

        bool active() const { return enabled && !entries.empty(); }
        i32 slots() const { return static_cast<i32>(entries.size()); }
        void disable() {
            enabled = false;
            entries.clear();
        }
    };
    static bool same_x_(ecref<vecXd> a, ecref<vecXd> b) {
        if (a.size() != b.size()) return false;
        return (a.array() == b.array()).all();
    }
    template <typename T>
    static CacheEntry<T>* find_slot_(CacheSet<T>& set) {
        if (!set.active()) return nullptr;
        for (auto& entry : set.entries) {
            if (!entry.has_value) return &entry; // return empty slot to use
        }
        CacheEntry<T>* lru = &set.entries.front(); // create entry if none empty
        for (auto& entry : set.entries) {          // replace oldest entry
            if (entry.lru_tick < lru->lru_tick) lru = &entry;
        }
        return lru;
    }
    static bool cache_lookup_(CacheSet<f64>& set, ecref<vecXd> x, f64& out) {
        if (!set.active()) {
            ++set.misses; // set as miss if cache is inactive
            return false;
        }
        for (auto& entry : set.entries) {
            if (!entry.has_value || !same_x_(entry.x, x)) continue;
            entry.lru_tick = ++set.tick;
            out = entry.value;
            ++set.hits;
            return true;
        }
        ++set.misses;
        return false;
    }
    static bool cache_lookup_(CacheSet<vecXd>& set, ecref<vecXd> x, eref<vecXd> out) {
        if (!set.active()) {
            ++set.misses; // set as miss if cache is inactive
            return false;
        }
        for (auto& entry : set.entries) {
            if (!entry.has_value || !same_x_(entry.x, x)) continue;
            entry.lru_tick = ++set.tick;
            out = entry.value;
            ++set.hits;
            return true;
        }
        ++set.misses;
        return false;
    }
    static bool cache_lookup_(CacheSet<matXd>& set, ecref<vecXd> x, eref<matXd> out) {
        if (!set.active()) {
            ++set.misses;
            return false;
        }
        for (auto& entry : set.entries) {
            if (!entry.has_value || !same_x_(entry.x, x)) continue;
            entry.lru_tick = ++set.tick;
            out = entry.value;
            ++set.hits;
            return true;
        }
        ++set.misses;
        return false;
    }
    static void cache_store_(CacheSet<f64>& set, ecref<vecXd> x, f64 value) {
        CacheEntry<f64>* slot = find_slot_(set);
        if (!slot) return;
        slot->x = x;
        slot->value = value;
        slot->has_value = true;
        slot->lru_tick = ++set.tick;
    }
    static void cache_store_(CacheSet<vecXd>& set, ecref<vecXd> x, ecref<vecXd> value) {
        CacheEntry<vecXd>* slot = find_slot_(set);
        if (!slot) return;
        slot->x = x;
        slot->value = value;
        slot->has_value = true;
        slot->lru_tick = ++set.tick;
    }
    static void cache_store_(CacheSet<matXd>& set, ecref<vecXd> x, ecref<matXd> value) {
        CacheEntry<matXd>* slot = find_slot_(set);
        if (!slot) return;
        slot->x = x;
        slot->value = value;
        slot->has_value = true;
        slot->lru_tick = ++set.tick;
    }
    void maybe_apply_hessian_guard_(i32 n) {
        if (!opt_.cache.enabled) return;
        if (!opt_.cache.enforce_max_bytes) return;
        if (opt_.cache.max_bytes < 0) return;
        if (!h_cache_.active()) return;

        const i64 needed = cache_bytes(n, h_cache_.slots());
        if (needed <= opt_.cache.max_bytes) return;

        h_cache_.disable();
    }
    bool can_eval_f_() const {
        return !limit_enabled(opt_.limits.max_f_evals)
               || (f_evals_ < opt_.limits.max_f_evals);
    }
    bool can_eval_g_() const {
        return !limit_enabled(opt_.limits.max_g_evals)
               || (g_evals_ < opt_.limits.max_g_evals);
    }
    bool can_eval_h_() const {
        return !limit_enabled(opt_.limits.max_h_evals)
               || (h_evals_ < opt_.limits.max_h_evals);
    }

  private:
    const Obj& obj_;
    const Options& opt_;

    // evaluation counters
    i32 f_evals_ = 0;
    i32 g_evals_ = 0;
    i32 h_evals_ = 0;
    i32 hv_evals_ = 0;

    // cache
    CacheSet<f64> f_cache_;
    CacheSet<vecXd> g_cache_;
    CacheSet<matXd> h_cache_;
    matXd hv_H_; // temp to avoid reallocating
};

} // namespace sOPT