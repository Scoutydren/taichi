#include "taichi/program/aot_module_builder.h"

#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend(identifier, kernel);
}

void AotModuleBuilder::add_field(const std::string &identifier,
<<<<<<< HEAD
                                 bool is_scalar,
                                 DataType dt,
                                 std::pair<int, int> shape,
                                 int vector_size) {
  add_per_backend_field(identifier, is_scalar, dt, shape, vector_size);
=======
                                 bool is_vector,
                                 DataType dt,
                                 std::pair<int, int> shape,
                                 int vector_size) {
  add_per_backend_field(identifier, is_vector, dt, shape, vector_size);
>>>>>>> 57c1a18f (Auto Format)
}

void AotModuleBuilder::add_kernel_template(const std::string &identifier,
                                           const std::string &key,
                                           Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend_tmpl(identifier, key, kernel);
}

}  // namespace lang
}  // namespace taichi
