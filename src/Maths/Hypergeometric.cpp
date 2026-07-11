#include "Hypergeometric.hpp"
#include <cassert>
#include <cmath>
#include <complex>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_result.h>
#include <type_traits>

#if defined(AMPSCI_USE_FLINT3) || defined(AMPSCI_USE_FLINT2)
// FLINT >= 3.0 merged Arb in: headers live under flint/ and link with -lflint
// (AMPSCI_USE_FLINT3). FLINT 2.x keeps acb_hypgeom in the separate Arb library:
// top-level headers, link with -lflint-arb (Debian/Ubuntu) or -larb (upstream)
// (AMPSCI_USE_FLINT2); see configure.sh. The acb_*/arb_*/arf_* API used below is
// identical either way.
#ifdef AMPSCI_USE_FLINT2
#include <acb.h>
#include <acb_hypgeom.h>
#else
#include <flint/acb.h>
#include <flint/acb_hypgeom.h>
#endif
#endif

// Manuall switch off (for testing only) (doesn't change has_flint)
// #undef AMPSCI_USE_FLINT3
// #undef AMPSCI_USE_FLINT2

namespace Hypergeometric {

template <typename T>
T H1f1(T a, double b, T z, double s) {
  static_assert(std::is_same_v<T, double> ||
                  std::is_same_v<T, std::complex<double>>,
                "H1f1: T must be double or std::complex<double>");

  if constexpr (std::is_same_v<T, double>) {
    // Real case: GSL, in double precision
    // static: set once only (thread safe)
    [[maybe_unused]] static const auto hndl = gsl_set_error_handler_off();
    gsl_sf_result res;
    const auto status = gsl_sf_hyperg_1F1_e(a, b, z, &res);
    const auto val = status == 0 ? res.val : 0.0;
    return s == 0.0 ? val : std::exp(s) * val;
  } else {
#if defined(AMPSCI_USE_FLINT3) || defined(AMPSCI_USE_FLINT2)
    // Complex case: FLINT ball arithmetic; working precision is increased
    // until the result is accurate to (at least) full double precision
    // https://flintlib.org/doc/acb_hypgeom.html
    acb_t res, aa, bb, zz;
    acb_init(res);
    acb_init(aa);
    acb_init(bb);
    acb_init(zz);
    acb_set_d_d(aa, a.real(), a.imag());
    acb_set_d(bb, b);
    acb_set_d_d(zz, z.real(), z.imag());
    // Probably overkill, but following example ("arb"itrary precission):
    // https://arblib.org/using.html
    slong prec = 64;
    for (; prec <= 16384; prec *= 2) {
      acb_hypgeom_m(res, aa, bb, zz, 0, prec);
      if (acb_rel_accuracy_bits(res) >= 53) {
        break;
      }
    }
    if (s != 0.0) {
      // apply scale e^{s} at matching precision (no accuracy loss)
      arb_t scale;
      arb_init(scale);
      arb_set_d(scale, s);
      arb_exp(scale, scale, prec);
      acb_mul_arb(res, res, scale, prec);
      arb_clear(scale);
    }
    const auto re = arf_get_d(arb_midref(acb_realref(res)), ARF_RND_NEAR);
    const auto im = arf_get_d(arb_midref(acb_imagref(res)), ARF_RND_NEAR);
    acb_clear(res);
    acb_clear(aa);
    acb_clear(bb);
    acb_clear(zz);
    return {re, im};
#else
    // Stub, when compiled without FLINT:
    // Run-time failure" if ever called:
    assert(false && "Requires FLINT to call H1f1 with complex args. See docs");
    (void)a;
    (void)b;
    (void)z;
    (void)s;
    return {std::nan(""), std::nan("")};
#endif
  }
}

// explicit instantiations:
template double H1f1(double, double, double, double);
template std::complex<double> H1f1(std::complex<double>, double,
                                   std::complex<double>, double);

} // namespace Hypergeometric
