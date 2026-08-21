#pragma once
/*! @file
  @brief Include instead of `<omp.h>` to allow compilation with or without OpenMP.

  @details
  If OpenMP is not available, stub macros are provided so `#pragma omp`
  directives are silently ignored, and the common `omp_*thread` functions return
  0 or 1.
*/
#include <dlfcn.h>
#include <string>

#if defined(_OPENMP)
#include <omp.h>
namespace qip {
//! True if compiled with OpenMP support, false otherwise.
constexpr bool use_omp = true;
} // namespace qip
#else
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
namespace qip {
//! True if compiled with OpenMP support, false otherwise.
constexpr bool use_omp = false;
} // namespace qip
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#endif

namespace qip {
//! Returns a short string describing the threading status, e.g.
//! "Using OpenMP with 8 threads." or "Single-threaded."
inline std::string omp_details() {
  return use_omp ? "Using OpenMP with " +
                     std::to_string(omp_get_max_threads()) + " threads." :
                   "Single-threaded.";
}

//==============================================================================
/*!
  @brief Scoped guard: forces single-threaded BLAS (OpenBLAS) for the
  lifetime of the object; restores the previous BLAS thread count on
  destruction.

  @details
  BLAS calls made from inside an OpenMP parallel region contend badly with
  the pthread-build OpenBLAS thread pool (every concurrent call fights over
  the same pool and its global lock). The outer OpenMP parallelism already
  saturates the cores there, so BLAS should run single-threaded. Construct
  one of these in serial code just before entering such a region; the
  effect is strictly scoped to the object's lifetime.

  OpenBLAS is located at run time (dlsym), so there is no compile-time
  dependence on it: with any other BLAS library the guard is a no-op
  (reference BLAS is single-threaded anyway). Also a no-op when OpenMP is
  not compiled in, or when only one OpenMP thread is available: BLAS
  threading is then the only parallelism, and is left alone.
*/
class SingleThreadBlas {
public:
  SingleThreadBlas() {
    if constexpr (use_omp) {
      if (omp_get_max_threads() > 1 && oblas_get() && oblas_set()) {
        m_saved = oblas_get()();
        if (m_saved > 1) {
          oblas_set()(1);
        }
      }
    }
  }
  ~SingleThreadBlas() {
    if (m_saved > 1) {
      oblas_set()(m_saved);
    }
  }
  SingleThreadBlas(const SingleThreadBlas &) = delete;
  SingleThreadBlas &operator=(const SingleThreadBlas &) = delete;

private:
  int m_saved{0};
  using fn_set = void (*)(int);
  using fn_get = int (*)();
  // Resolved once; nullptr when the linked BLAS is not OpenBLAS
  static fn_set oblas_set() {
    static const auto f =
      reinterpret_cast<fn_set>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    return f;
  }
  static fn_get oblas_get() {
    static const auto f =
      reinterpret_cast<fn_get>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    return f;
  }
};
} // namespace qip
