from taichi.lang import impl, kernel_arguments, kernel_impl, expr, matrix


class Module:
    """An AOT module to save and load Taichi kernels.

    This module serializes the Taichi kernels for a specific arch. The
    serialized module can later be loaded to run on that backend, without the
    Python environment.

    Example::

        m = ti.aot.Module(ti.metal)
        m.add_kernel(foo)
        m.add_kernel(bar)

        m.save('/path/to/module')

        # Now the module file '/path/to/module' contains the Metal kernels
        # for running ``foo`` and ``bar``.
    """
    def __init__(self, arch):
        self._arch = arch
        self._kernels = []
        self._fields = {}
        impl.get_runtime().materialize()
        self._aot_builder = impl.get_runtime().prog.make_aot_module_builder(
            arch)
    
    def add_field(self, name, field):
      """Add a taichi field to the AOT module.
      Args: 
        name: name of taichi field
        field: taichi field

        a = ti.field("something")
        b = ti.field("something")

        m.add_field(a)
        m.add_field(b)
        
        Must add in sequence
      """
      # assert isinstance(field, expr.Expr)

      # TODO: pass field.shape, field.dt to aotModuleBuilder
      field_number = len(self._fields)
      print(field_number)
      is_vector = False
      self._fields[name] = field
      if type(field) is matrix.Matrix:
        assert isinstance(field, matrix.Matrix)
        is_vector = True
      else:
        assert isinstance(field, expr.Expr)
      self._aot_builder.add_field(name, is_vector)

    def add_kernel(self, kernel_fn, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.

        TODO:
          * Support external array
        """
        name = name or kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        for i in range(len(kernel.argument_annotations)):
            anno = kernel.argument_annotations[i]
            if isinstance(anno, kernel_arguments.ArgExtArray):
                raise RuntimeError('Arg type `ext_arr` not supported yet')
            else:
                # For primitive types, we can just inject a dummy value.
                injected_args.append(0)
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add(name, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)
    
    def add_kernel_template(self, kernel_fn, template_args):
        """Add a taichi kernel (with template parameters) to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.


        Example:
          Note that if `kernel_fn` contains at least one template parameter, it
          is required that users provide an explicit `name`. In addition, all
          the values of these template parameters must be instantiated via
          `template_args`.

          Usage::

            @ti.kernel
            def bar_tmpl(a: ti.template()):
              x = a
              # or y = a
              # do something with `x` or `y`

            m = ti.aot.Module(arch)
            m.add_kernel_template(bar_tmpl, template_args={'a': x})
            m.add_kernel_template(bar_tmpl, template_args={'a': y})

            @ti.kernel
            def bar_tmpl_multiple_args(a: ti.template(), b: ti.template())
              x = a
              y = b
              # do something with `x` and `y`
            
            m.add_kernel_template(bar_tmpl_multiple_args, template_args = {'a': x})

        TODO:
          * Support external array
        """
        name = kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = []
        key = ""
        for i in range(len(kernel.argument_annotations)):
            anno = kernel.argument_annotations[i]
            if isinstance(anno, kernel_arguments.Template):
                key += kernel.argument_names[i]
                value = template_args[kernel.argument_names[i]]
                for ky, val in self._fields.items():
                  if (val is value):
                    key += "=" + ky + "/"
                injected_args.append(value)
            else:
                injected_args.append(0)
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add_kernel_template(name, key, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)

    def save(self, filepath, filename):
        self._aot_builder.dump(filepath, filename)
