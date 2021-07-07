#pragma once

#include <string>

namespace taichi {
namespace lang {

class Kernel;

class AotModuleBuilder {
 public:
  virtual ~AotModuleBuilder() = default;

  void add(const std::string &identifier, Kernel *kernel);

  void add_field(const std::string &identifier, bool is_vector);

  void add_kernel_template(const std::string &identifier, 
                           const std::string &key, 
                           Kernel *kernel);

  virtual void dump(const std::string &output_dir,
                    const std::string &filename) const = 0;

 protected:
  /**
   * Intended to be overriden by each backend's implementation.
   */
  virtual void add_per_backend(const std::string &identifier,
                               Kernel *kernel) = 0;
  virtual void add_per_backend_field(const std::string &identifier, bool is_vector) = 0;
  virtual void add_per_backend_tmpl(const std::string &identifier, 
                                    const std::string &key, 
                                    Kernel *kernel) = 0;
};

}  // namespace lang
}  // namespace taichi
