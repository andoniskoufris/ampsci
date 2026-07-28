#include "version.hpp"
#include "Maths/Hypergeometric.hpp" // for has_flint
#include <cstdlib>
#include <dlfcn.h>
#include <gsl/gsl_version.h>
#include <string>

// Macro translates constants to "strings"
#define XSTRING(s) STRING(s)
#define STRING(s) #s

// Constants refer to git revision info.
// These are passed in at compile time via -D flag
// If not set, the code will still work (these will just be blank)
// These are in the .cpp file, so we only need to re-build this file (and
// re-link) whenever we want updated git version info [most applicable for the
// GITMODIFIED option, which is not easy to implement otherwise]
#ifndef GITBRANCH
#define GITBRANCH
#endif
#ifndef GITREVISION
#define GITREVISION
#endif
#ifndef GITMODIFIED
#define GITMODIFIED
#endif
#ifndef CXXVERSION
#define CXXVERSION
#endif
#ifndef COMPTIME
#define COMPTIME
#endif
#ifndef EXTERNAL_MODULES
#define EXTERNAL_MODULES
#endif
// GSL_VERSION defined by GSL library (if it exists)
#ifndef GSL_VERSION
#define GSL_VERSION ""
#endif
// _OPENMP defined by openMP library (if it exists)
#ifdef _OPENMP
#define OMP_VERSION _OPENMP
#else
#define OMP_VERSION 0
#endif

//==============================================================================
namespace version {

// git branch for current compilation (if available)
static const std::string git_branch = XSTRING(GITBRANCH);
// git hash for current compilation (if available)
static const std::string git_revision = XSTRING(GITREVISION);
// List of files that have been modified since last git commit
static const std::string git_modified = XSTRING(GITMODIFIED);
// Compiler version information
static const std::string cxx_version = XSTRING(CXXVERSION);
// Date and time of compilation
static const std::string compiled_time = XSTRING(COMPTIME);
// External modules compiled in (space-separated filenames, may be empty)
static const std::string external_modules = XSTRING(EXTERNAL_MODULES);
// ampsci version
static const std::string ampsci_version = std::string(AMPSCI_VERSION);
// gsl library version
static const std::string gsl_version = std::string(GSL_VERSION);
// OpenMP library version
static const std::string omp_version = XSTRING(OMP_VERSION);

std::string version() {
  std::string v = git_revision.empty() ? ampsci_version :
                  git_modified.empty() ? ampsci_version + " [" + git_branch +
                                           "/" + git_revision + "]" :
                                         ampsci_version + " [" + git_branch +
                                           "/" + git_revision + "]*\n" +
                                           " *(Modified: " + git_modified + ")";
  if (!external_modules.empty())
    v += "\n External modules: " + external_modules;
  return v;
}

std::string compiled() { return cxx_version + " " + compiled_time; }

// GSL version we compiled against, compared to the library actually loaded.
// gsl_version (unqualified) is the compile-time GSL_VERSION macro from the
// header; ::gsl_version is a global exported by libgsl itself. If these
// disagree, the header and library are mismatched and behaviour is undefined
static std::string gsl_info() {
  const std::string loaded{::gsl_version};
  if (gsl_version == loaded)
    return gsl_version;
  return gsl_version + " (compiled) does not match " + loaded + " (loaded!)";
}

// FLINT is optional; it provides the complex hypergeometric functions.
// Reports whether it was compiled in, and the version actually loaded.
// flint_version is a global exported by libflint, found here with dlsym so we
// do not need the FLINT headers (which are only available in some builds).
// Note it is declared 'char flint_version[]', an array, so dlsym returns the
// address of the string itself; do not dereference it (unlike ::gsl_version,
// which is a pointer)
static std::string flint_info() {
  if (!Hypergeometric::has_flint)
    return "not compiled in (optional)";
  const auto symbol = dlsym(RTLD_DEFAULT, "flint_version");
  if (symbol == nullptr)
    return "compiled in (version unknown)";
  return static_cast<const char *>(symbol);
}

static std::string blas_info() {
  using fn_str = const char *(*)();
  using fn_mkl = void (*)(char *, int);
  auto oblas_cfg =
    reinterpret_cast<fn_str>(dlsym(RTLD_DEFAULT, "openblas_get_config"));
  if (oblas_cfg)
    return std::string("OpenBLAS: ") + oblas_cfg();
  auto mkl_ver =
    reinterpret_cast<fn_mkl>(dlsym(RTLD_DEFAULT, "MKL_Get_Version_String"));
  if (mkl_ver) {
    char buf[256] = {};
    mkl_ver(buf, 256);
    return std::string("Intel MKL: ") + buf;
  }
  return "Reference LAPACK/BLAS";
}

// Full path of the library that actually provides symbol at run time.
// This is not always the one named on the link line: libgslcblas and openblas
// both define cblas_dgemm, and whichever is linked first wins.
// One symbol is enough per family: an implementation defines the whole cblas_*
// set, or the whole LAPACK set, so the linker picks one library for each.
static std::string symbol_provider(const char *symbol) {
  const auto address = dlsym(RTLD_DEFAULT, symbol);
  if (address == nullptr)
    return "not found";
  Dl_info info;
  if (dladdr(address, &info) == 0 || info.dli_fname == nullptr)
    return "unknown";
  // resolve any symlinks: e.g., libblas.so.3 may point to openblas or reference
  char *resolved = realpath(info.dli_fname, nullptr);
  if (resolved == nullptr)
    return info.dli_fname;
  const std::string path{resolved};
  std::free(resolved);
  return path;
}

std::string blas_threads() {
  using fn_int = int (*)();
  auto oblas_thr =
    reinterpret_cast<fn_int>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
  if (oblas_thr)
    return "OpenBLAS: " + std::to_string(oblas_thr()) + " threads.";
  auto mkl_thr =
    reinterpret_cast<fn_int>(dlsym(RTLD_DEFAULT, "mkl_get_max_threads"));
  if (mkl_thr)
    return "MKL: " + std::to_string(mkl_thr()) + " threads.";
  return "";
}

std::string libraries() {
  return "  GSL (GNU Scientific Libraries): " + gsl_info() + '\n' +
         "  OpenMP: " + omp_version + '\n' + "  LAPACK/BLAS: " + blas_info() +
         '\n' + "    cblas from: " + symbol_provider("cblas_dgemm") + '\n' +
         "    lapack from: " + symbol_provider("dgetrf_") + '\n' +
         "  FLINT: " + flint_info();
}

} // namespace version
