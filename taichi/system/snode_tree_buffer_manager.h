#pragma once
#include "taichi/llvm/llvm_context.h"
#include "taichi/inc/constants.h"
#include "taichi/struct/snode_tree.h"
#define TI_RUNTIME_HOST

#include <set>

using Ptr_ti = uint8_t *;

TLANG_NAMESPACE_BEGIN

class Program;

class SNodeTreeBufferManager {
 public:
  SNodeTreeBufferManager(Program *prog);

  void merge_and_insert(Ptr_ti ptr, std::size_t size);

  Ptr_ti allocate(JITModule *runtime_jit,
               void *runtime,
               std::size_t size,
               std::size_t alignment,
               const int snode_tree_id);

  void destroy(SNodeTree *snode_tree);

 private:
  std::set<std::pair<std::size_t, Ptr_ti>> size_set_;
  std::map<Ptr_ti, std::size_t> ptr_map_;
  Program *prog_;
  Ptr_ti roots_[taichi_max_num_snode_trees];
  std::size_t sizes_[taichi_max_num_snode_trees];
};

TLANG_NAMESPACE_END
