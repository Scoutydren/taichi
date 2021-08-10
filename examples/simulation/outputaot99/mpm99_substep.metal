#include <metal_stdlib>
#include <metal_compute>
using namespace metal;
namespace {
using byte = char;

template <typename T, typename G> T union_cast(G g) { static_assert(sizeof(T) == sizeof(G), "Size mismatch"); return *reinterpret_cast<thread const T *>(&g); } inline int ifloordiv(int lhs, int rhs) { const int intm = (lhs / rhs); return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs)) ? (intm - 1) : intm); } int32_t pow_i32(int32_t x, int32_t n) { int32_t tmp = x; int32_t ans = 1; while (n) { if (n & 1) ans *= tmp; tmp *= tmp; n >>= 1; } return ans; } float fatomic_fetch_add(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val + operand); ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_min(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val < operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_max(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val > operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } struct RandState { uint32_t seed; }; uint32_t metal_rand_u32(device RandState * state) { device uint *sp = (device uint *)&(state->seed); bool done = false; uint32_t nxt = 0; while (!done) { uint32_t o = *sp; nxt = o * 1103515245 + 12345; done = atomic_compare_exchange_weak_explicit( (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed, metal::memory_order_relaxed); } return nxt * 1000000007; } int32_t metal_rand_i32(device RandState * state) { return metal_rand_u32(state); } float metal_rand_f32(device RandState *state) { return metal_rand_u32(state) * (1.0f / 4294967296.0f); }

constant constexpr int kTaichiMaxNumIndices = 8; constant constexpr int kTaichiNumChunks = 1024; constant constexpr int kAlignment = 8; using PtrOffset = int32_t; struct MemoryAllocator { atomic_int next; constant constexpr static int kInitOffset = 8; static inline bool is_valid(PtrOffset v) { return v >= kInitOffset; } }; struct ListManagerData { int32_t element_stride = 0; int32_t log2_num_elems_per_chunk = 0; atomic_int next; atomic_int chunks[kTaichiNumChunks]; struct ReservedElemPtrOffset { public: ReservedElemPtrOffset() = default; explicit ReservedElemPtrOffset(PtrOffset v) : val_(v) { } inline bool is_valid() const { return is_valid(val_); } inline static bool is_valid(PtrOffset v) { return MemoryAllocator::is_valid(v); } inline PtrOffset value() const { return val_; } private: PtrOffset val_{0}; }; }; struct NodeManagerData { using ElemIndex = ListManagerData::ReservedElemPtrOffset; ListManagerData data_list; ListManagerData free_list; ListManagerData recycled_list; atomic_int free_list_used; int recycled_list_size_backup; }; struct SNodeMeta { enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3, Pointer = 4, BitStruct = 5, }; int32_t element_stride = 0; int32_t num_slots = 0; int32_t mem_offset_in_parent = 0; int32_t type = 0; }; struct SNodeExtractors { struct Extractor { int32_t start = 0; int32_t num_bits = 0; int32_t acc_offset = 0; int32_t num_elements_from_root = 0; }; Extractor extractors[kTaichiMaxNumIndices]; }; struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; }; struct ListgenElement { ElementCoords coords; int32_t mem_offset = 0; struct BelongedNodeManager { int32_t id = -1; NodeManagerData::ElemIndex elem_idx; }; BelongedNodeManager belonged_nodemgr; inline bool in_root_buffer() const { return belonged_nodemgr.id < 0; } };

struct Runtime {
  SNodeMeta snode_metas[25];
  SNodeExtractors snode_extractors[25];
  ListManagerData snode_lists[25];
  NodeManagerData snode_allocators[25];
  NodeManagerData::ElemIndex ambient_indices[25];
  uint32_t rand_seeds[65536];
};

[[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator *ma, int32_t size) { size = ((size + kAlignment - 1) / kAlignment) * kAlignment; return atomic_fetch_add_explicit(&ma->next, size, metal::memory_order_relaxed); } [[maybe_unused]] device char *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) { return reinterpret_cast<device char *>(ma + 1) + offs; } struct ListManager { using ReservedElemPtrOffset = ListManagerData::ReservedElemPtrOffset; device ListManagerData *lm_data; device MemoryAllocator *mem_alloc; inline int num_active() { return atomic_load_explicit(&(lm_data->next), metal::memory_order_relaxed); } inline void resize(int sz) { atomic_store_explicit(&(lm_data->next), sz, metal::memory_order_relaxed); } inline void clear() { resize(0); } ReservedElemPtrOffset reserve_new_elem() { const int elem_idx = atomic_fetch_add_explicit( &lm_data->next, 1, metal::memory_order_relaxed); const int chunk_idx = get_chunk_index(elem_idx); const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx); const auto offset = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return ReservedElemPtrOffset{offset}; } device char *append() { auto reserved = reserve_new_elem(); return get_ptr(reserved); } template <typename T> void append(thread const T &elem) { device char *ptr = append(); thread char *elem_ptr = (thread char *)(&elem); for (int i = 0; i < lm_data->element_stride; ++i) { *ptr = *elem_ptr; ++ptr; ++elem_ptr; } } device char *get_ptr(ReservedElemPtrOffset offs) { return mtl_memalloc_to_ptr(mem_alloc, offs.value()); } device char *get_ptr(int i) { const int chunk_idx = get_chunk_index(i); const PtrOffset chunk_ptr_offs = atomic_load_explicit( lm_data->chunks + chunk_idx, metal::memory_order_relaxed); return get_elem_from_chunk(i, chunk_ptr_offs); } template <typename T> T get(int i) { return *reinterpret_cast<device T *>(get_ptr(i)); } private: inline int get_chunk_index(int elem_idx) const { return elem_idx >> lm_data->log2_num_elems_per_chunk; } PtrOffset ensure_chunk(int chunk_idx) { PtrOffset offs = 0; const int chunk_bytes = (lm_data->element_stride << lm_data->log2_num_elems_per_chunk); while (true) { int stored = 0; const bool is_me = atomic_compare_exchange_weak_explicit( lm_data->chunks + chunk_idx, &stored, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes); atomic_store_explicit(lm_data->chunks + chunk_idx, offs, metal::memory_order_relaxed); break; } else if (stored > 1) { offs = stored; break; } } return offs; } PtrOffset get_elem_ptr_offs_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1); return chunk_ptr_offs + ((elem_idx & mask) * lm_data->element_stride); } device char *get_elem_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const auto offs = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return mtl_memalloc_to_ptr(mem_alloc, offs); } }; struct NodeManager { using ElemIndex = NodeManagerData::ElemIndex; device NodeManagerData *nm_data; device MemoryAllocator *mem_alloc; ElemIndex allocate() { ListManager free_list; free_list.lm_data = &(nm_data->free_list); free_list.mem_alloc = mem_alloc; ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; const int cur_used = atomic_fetch_add_explicit( &(nm_data->free_list_used), 1, metal::memory_order_relaxed); if (cur_used < free_list.num_active()) { return free_list.get<ElemIndex>(cur_used); } return data_list.reserve_new_elem(); } device byte *get(ElemIndex i) { ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; return data_list.get_ptr(i); } void recycle(ElemIndex i) { ListManager recycled_list; recycled_list.lm_data = &(nm_data->recycled_list); recycled_list.mem_alloc = mem_alloc; recycled_list.append(i); } }; class SNodeRep_dense { public: void init(device byte * addr) { addr_ = addr; } inline device byte *addr() { return addr_; } inline bool is_active(int) { return true; } inline void activate(int) { } inline void deactivate(int) { } private: device byte *addr_ = nullptr; }; using SNodeRep_root = SNodeRep_dense; class SNodeRep_bitmasked { public: constant static constexpr int kBitsPerMask = (sizeof(uint32_t) * 8); void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { device auto *ptr = to_bitmask_ptr(i); uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ((bits >> (i % kBitsPerMask)) & 1); } void activate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = (1 << (i % kBitsPerMask)); atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed); } void deactivate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = ~(1 << (i % kBitsPerMask)); atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed); } private: inline device atomic_uint *to_bitmask_ptr(int i) { return reinterpret_cast<device atomic_uint *>(addr_ + meta_offset_) + (i / kBitsPerMask); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_dynamic { public: void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { const auto n = atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); return i < n; } void activate(int i) { device auto *ptr = to_meta_ptr(); atomic_fetch_max_explicit(ptr, (i + 1), metal::memory_order_relaxed); return; } void deactivate() { device auto *ptr = to_meta_ptr(); atomic_store_explicit(ptr, 0, metal::memory_order_relaxed); } int append(int32_t data) { device auto *ptr = to_meta_ptr(); int me = atomic_fetch_add_explicit(ptr, 1, metal::memory_order_relaxed); *(reinterpret_cast<device int32_t *>(addr_) + me) = data; return me; } int length() { return atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); } private: inline device atomic_int *to_meta_ptr() { return reinterpret_cast<device atomic_int *>(addr_ + meta_offset_); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_pointer { public: using ElemIndex = NodeManagerData::ElemIndex; void init(device byte * addr, NodeManager nm, ElemIndex ambient_idx) { addr_ = addr; nm_ = nm; ambient_idx_ = ambient_idx; } device byte *child_or_ambient_addr(int i) { auto nm_idx = to_nodemgr_idx(addr_, i); nm_idx = nm_idx.is_valid() ? nm_idx : ambient_idx_; return nm_.get(nm_idx); } inline bool is_active(int i) { return is_active(addr_, i); } void activate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); auto nm_idx_val = atomic_load_explicit(nm_idx_ptr, metal::memory_order_relaxed); while (!ElemIndex::is_valid(nm_idx_val)) { nm_idx_val = 0; const bool is_me = atomic_compare_exchange_weak_explicit( nm_idx_ptr, &nm_idx_val, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { nm_idx_val = nm_.allocate().value(); atomic_store_explicit(nm_idx_ptr, nm_idx_val, metal::memory_order_relaxed); break; } else if (ElemIndex::is_valid(nm_idx_val)) { break; } } } void deactivate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); const auto old_nm_idx_val = atomic_exchange_explicit( nm_idx_ptr, 0, metal::memory_order_relaxed); const auto old_nm_idx = ElemIndex(old_nm_idx_val); if (!old_nm_idx.is_valid()) { return; } nm_.recycle(old_nm_idx); } static inline device atomic_int *to_nodemgr_idx_ptr(device byte * addr, int ch_i) { return reinterpret_cast<device atomic_int *>(addr + ch_i * sizeof(ElemIndex)); } static inline ElemIndex to_nodemgr_idx(device byte * addr, int ch_i) { device auto *ptr = to_nodemgr_idx_ptr(addr, ch_i); const auto v = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ElemIndex(v); } static bool is_active(device byte * addr, int ch_i) { return to_nodemgr_idx(addr, ch_i).is_valid(); } private: device byte *addr_; NodeManager nm_; ElemIndex ambient_idx_; }; [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return true; } else if (meta.type == SNodeMeta::Dynamic) { SNodeRep_dynamic rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Bitmasked) { SNodeRep_bitmasked rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Pointer) { return SNodeRep_pointer::is_active(addr, i); } return false; } [[maybe_unused]] void refine_coordinates( thread const ElementCoords &parent, device const SNodeExtractors &child_extrators, int l, thread ElementCoords *child) { for (int i = 0; i < kTaichiMaxNumIndices; ++i) { device const auto &ex = child_extrators.extractors[i]; const int mask = ((1 << ex.num_bits) - 1); const int addition = ((l >> ex.acc_offset) & mask); child->at[i] = ((parent.at[i] << ex.num_bits) | addition); } } [[maybe_unused]] device byte *mtl_lgen_snode_addr( thread const ListgenElement &lgen, device byte *root_addr, device Runtime *rtm, device MemoryAllocator *mem_alloc) { if (lgen.in_root_buffer()) { return root_addr + lgen.mem_offset; } NodeManager nm; nm.nm_data = (rtm->snode_allocators + lgen.belonged_nodemgr.id); nm.mem_alloc = mem_alloc; device byte *addr = nm.get(lgen.belonged_nodemgr.elem_idx); return addr + lgen.mem_offset; } [[maybe_unused]] void run_gc_compact_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, const int tid, const int grid_size) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_load_explicit(&(nm.nm_data->free_list_used), metal::memory_order_relaxed); int num_to_copy = 0; if (free_used * 2 > free_size) { num_to_copy = free_size - free_used; } else { num_to_copy = free_used; } const int offs = free_size - num_to_copy; using ElemIndex = NodeManager::ElemIndex; for (int ii = tid; ii < num_to_copy; ii += grid_size) { device auto *dest = reinterpret_cast<device ElemIndex *>(free_list.get_ptr(ii)); *dest = free_list.get<ElemIndex>(ii + offs); } } [[maybe_unused]] void run_gc_reset_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_exchange_explicit( &(nm.nm_data->free_list_used), 0, metal::memory_order_relaxed); int free_remaining = free_size - free_used; free_remaining = free_remaining > 0 ? free_remaining : 0; free_list.resize(free_remaining); nm.nm_data->recycled_list_size_backup = atomic_exchange_explicit( &(nm.nm_data->recycled_list.next), 0, metal::memory_order_relaxed); } struct GCMoveRecycledToFreeThreadParams { int thread_position_in_threadgroup; int threadgroup_position_in_grid; int threadgroups_per_grid; int threads_per_threadgroup; }; [[maybe_unused]] void run_gc_move_recycled_to_free( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, thread const GCMoveRecycledToFreeThreadParams &thparams) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; ListManager recycled_list; recycled_list.lm_data = &(nm.nm_data->recycled_list); recycled_list.mem_alloc = nm.mem_alloc; ListManager data_list; data_list.lm_data = &(nm.nm_data->data_list); data_list.mem_alloc = nm.mem_alloc; const int kInt32Stride = sizeof(int32_t); const int recycled_size = nm.nm_data->recycled_list_size_backup; using ElemIndex = NodeManager::ElemIndex; for (int ii = thparams.threadgroup_position_in_grid; ii < recycled_size; ii += thparams.threadgroups_per_grid) { const auto elem_idx = recycled_list.get<ElemIndex>(ii); device char *ptr = nm.get(elem_idx); device const char *ptr_end = ptr + data_list.lm_data->element_stride; const int ptr_mod = ((int64_t)(ptr) % kInt32Stride); if (ptr_mod) { device char *new_ptr = ptr + kInt32Stride - ptr_mod; if (thparams.thread_position_in_threadgroup == 0) { for (device char *p = ptr; p < new_ptr; ++p) { *p = 0; } } ptr = new_ptr; } ptr += (thparams.thread_position_in_threadgroup * kInt32Stride); while ((ptr + kInt32Stride) <= ptr_end) { *reinterpret_cast<device int32_t *>(ptr) = 0; ptr += (kInt32Stride * thparams.threads_per_threadgroup); } while (ptr < ptr_end) { *ptr = 0; ++ptr; } if (thparams.thread_position_in_threadgroup == 0) { free_list.append(elem_idx); } } }

struct SNodeBitPointer { device uint32_t *base; uint32_t offset; SNodeBitPointer(device byte * b, uint32_t o) : base((device uint32_t *)b), offset(o) { } }; template <typename C> C mtl_float_to_custom_int(float f) { const int32_t delta_bits = (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f); const float delta = union_cast<float>(delta_bits); return static_cast<C>(f + delta); } void mtl_set_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); bool ok = false; while (!ok) { P old_val = *(bp.base); P new_val = (old_val & (~mask)) | (value << bp.offset); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } } void mtl_set_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); atomic_store_explicit(atm_ptr, value, metal::memory_order_relaxed); } uint32_t mtl_atomic_add_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); P old_val = 0; bool ok = false; while (!ok) { old_val = *(bp.base); P new_val = old_val + (value << bp.offset); new_val = (old_val & (~mask)) | (new_val & mask); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } uint32_t mtl_atomic_add_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); return atomic_fetch_add_explicit(atm_ptr, value, metal::memory_order_relaxed); } namespace detail { template <bool Signed> struct SHRSelector { using type = int32_t; }; template <> struct SHRSelector<false> { using type = uint32_t; }; } template <typename C> C mtl_get_partial_bits(SNodeBitPointer bp, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const P phy_val = *(bp.base); using CSel = typename detail::SHRSelector<is_signed<C>::value>::type; const auto step1 = static_cast<CSel>(phy_val << (N - (bp.offset + bits))); return static_cast<C>(step1 >> (N - bits)); } template <typename C> C mtl_get_full_bits(SNodeBitPointer bp) { return static_cast<C>(*(bp.base)); }




struct S25 {
  // place
  constant static constexpr int stride = sizeof(float);

  S25(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S24_ch {
 public:
  S24_ch(device byte *a) : addr_(a) {}
  S25 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S25::stride;
 private:
  device byte *addr_;
};

struct S24 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S24_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S24(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S24_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S23 {
  // place
  constant static constexpr int stride = sizeof(float);

  S23(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S22 {
  // place
  constant static constexpr int stride = sizeof(float);

  S22(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S21_ch {
 public:
  S21_ch(device byte *a) : addr_(a) {}
  S22 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S23 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S22::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S22::stride + S23::stride;
 private:
  device byte *addr_;
};

struct S21 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S21_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S21(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S21_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S20 {
  // place
  constant static constexpr int stride = sizeof(float);

  S20(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S19_ch {
 public:
  S19_ch(device byte *a) : addr_(a) {}
  S20 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S20::stride;
 private:
  device byte *addr_;
};

struct S19 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S19_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S19(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S19_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S18 {
  // place
  constant static constexpr int stride = sizeof(int32_t);

  S18(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device int32_t*)v) {}

  device int32_t *val;
};

class S17_ch {
 public:
  S17_ch(device byte *a) : addr_(a) {}
  S18 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S18::stride;
 private:
  device byte *addr_;
};

struct S17 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S17_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S17(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S17_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S16 {
  // place
  constant static constexpr int stride = sizeof(float);

  S16(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S15 {
  // place
  constant static constexpr int stride = sizeof(float);

  S15(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S14 {
  // place
  constant static constexpr int stride = sizeof(float);

  S14(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S13 {
  // place
  constant static constexpr int stride = sizeof(float);

  S13(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S12_ch {
 public:
  S12_ch(device byte *a) : addr_(a) {}
  S13 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S14 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S13::stride), rtm, ma};
  }

  S15 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S13::stride + S14::stride), rtm, ma};
  }

  S16 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S13::stride + S14::stride + S15::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S13::stride + S14::stride + S15::stride + S16::stride;
 private:
  device byte *addr_;
};

struct S12 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S12_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S12(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S12_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S11 {
  // place
  constant static constexpr int stride = sizeof(float);

  S11(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S10 {
  // place
  constant static constexpr int stride = sizeof(float);

  S10(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S9 {
  // place
  constant static constexpr int stride = sizeof(float);

  S9(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S8 {
  // place
  constant static constexpr int stride = sizeof(float);

  S8(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S7_ch {
 public:
  S7_ch(device byte *a) : addr_(a) {}
  S8 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S9 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S8::stride), rtm, ma};
  }

  S10 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S8::stride + S9::stride), rtm, ma};
  }

  S11 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S8::stride + S9::stride + S10::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S8::stride + S9::stride + S10::stride + S11::stride;
 private:
  device byte *addr_;
};

struct S7 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S7_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S7(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S7_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S6 {
  // place
  constant static constexpr int stride = sizeof(float);

  S6(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S5 {
  // place
  constant static constexpr int stride = sizeof(float);

  S5(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S4_ch {
 public:
  S4_ch(device byte *a) : addr_(a) {}
  S5 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S6 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S5::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S5::stride + S6::stride;
 private:
  device byte *addr_;
};

struct S4 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S4_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S4(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S4_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S3 {
  // place
  constant static constexpr int stride = sizeof(float);

  S3(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S2 {
  // place
  constant static constexpr int stride = sizeof(float);

  S2(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S1_ch {
 public:
  S1_ch(device byte *a) : addr_(a) {}
  S2 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S3 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S2::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S2::stride + S3::stride;
 private:
  device byte *addr_;
};

struct S1 {
  // dense
  constant static constexpr int n = 16384;
  constant static constexpr int elem_stride = S1_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S1(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S1_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};

class S0_ch {
 public:
  S0_ch(device byte *a) : addr_(a) {}
  S1 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S4 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride), rtm, ma};
  }

  S7 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride), rtm, ma};
  }

  S12 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride), rtm, ma};
  }

  S17 get4(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride), rtm, ma};
  }

  S19 get5(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride + S17::stride), rtm, ma};
  }

  S21 get6(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride + S17::stride + S19::stride), rtm, ma};
  }

  S24 get7(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride + S17::stride + S19::stride + S21::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S1::stride + S4::stride + S7::stride + S12::stride + S17::stride + S19::stride + S21::stride + S24::stride;
 private:
  device byte *addr_;
};

struct S0 {
  // root
  constant static constexpr int n = 1;
  constant static constexpr int elem_stride = S0_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S0(device byte *addr) {
    rep_.init(addr);
  }

  S0_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_root rep_;
};



using AdStackPtr = thread byte *; inline thread uint32_t * mtl_ad_stack_n(AdStackPtr stack) { return reinterpret_cast<thread uint32_t *>(stack); } inline AdStackPtr mtl_ad_stack_data(AdStackPtr stack) { return stack + sizeof(uint32_t); } inline void mtl_ad_stack_init(AdStackPtr stack) { *mtl_ad_stack_n(stack) = 0; } inline AdStackPtr mtl_ad_stack_top_primal(AdStackPtr stack, int element_size) { const auto n = *mtl_ad_stack_n(stack); return mtl_ad_stack_data(stack) + (n - 1) * 2 * element_size; } inline AdStackPtr mtl_ad_stack_top_adjoint(AdStackPtr stack, int element_size) { return mtl_ad_stack_top_primal(stack, element_size) + element_size; } inline void mtl_ad_stack_pop(AdStackPtr stack) { thread auto &n = *mtl_ad_stack_n(stack); --n; } void mtl_ad_stack_push(AdStackPtr stack, int element_size) { thread auto &n = *mtl_ad_stack_n(stack); ++n; AdStackPtr data = mtl_ad_stack_top_primal(stack, element_size); for (int i = 0; i < element_size * 2; ++i) { data[i] = 0; } }

constant constexpr int kMetalNumBitsPerPrintMsgType = 4; constant constexpr int kMetalNumPrintMsgTypePerI32 = sizeof(int32_t) * 8 / kMetalNumBitsPerPrintMsgType; constant constexpr int kMetalPrintMsgTypeWidthMask = ((1 << kMetalNumBitsPerPrintMsgType) - 1); [[maybe_unused]] constexpr inline int mtl_compute_num_print_msg_typemasks( int num_entries) { return (num_entries + kMetalNumPrintMsgTypePerI32 - 1) / kMetalNumPrintMsgTypePerI32; } [[maybe_unused]] constexpr inline int mtl_compute_print_msg_bytes( int num_entries) { const int sz = sizeof(int32_t) * (1 + mtl_compute_num_print_msg_typemasks(num_entries) + num_entries); return sz; } class PrintMsg { public: enum Type { I32 = 1, U32 = 2, F32 = 3, Str = 4 }; PrintMsg(device int32_t *buf, int num_entries) : mask_buf_(buf), data_buf_(buf + mtl_compute_num_print_msg_typemasks(num_entries)) { } void pm_set_i32(int i, int x) { set_entry(i, x, Type::I32); } void pm_set_u32(int i, uint x) { const int32_t ix = static_cast<int32_t>(x); set_entry(i, ix, Type::U32); } void pm_set_f32(int i, float x) { const int32_t ix = *reinterpret_cast<thread int32_t *>(&x); set_entry(i, ix, Type::F32); } void pm_set_str(int i, int str_id) { set_entry(i, str_id, Type::Str); } Type pm_get_type(int i) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = mask_buf_[mask_i]; mask >>= typemask_shift(i_in_mask); mask &= kMetalPrintMsgTypeWidthMask; return (Type)mask; } int32_t pm_get_data(int i) { return data_buf_[i]; } private: void set_entry(int i, int32_t x, Type ty) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = ((int)ty & kMetalPrintMsgTypeWidthMask); mask <<= typemask_shift(i_in_mask); mask_buf_[mask_i] |= mask; data_buf_[i] = x; } inline static int typemask_shift(int i_in_mask) { return (kMetalNumPrintMsgTypePerI32 - 1 - i_in_mask) * kMetalNumBitsPerPrintMsgType; } device int32_t *mask_buf_; device int32_t *data_buf_; }; struct AssertRecorderData { atomic_int flag; int32_t num_args; }; class AssertRecorder { public: explicit AssertRecorder(device byte * addr) : ac_(reinterpret_cast<device AssertRecorderData *>(addr)) { } bool mark_first_failure() { return atomic_exchange_explicit(&(ac_->flag), 1, metal::memory_order_relaxed) == 0; } void set_num_args(int n) { ac_->num_args = n; } device int32_t *msg_buf_addr() { return reinterpret_cast<device int32_t *>(ac_ + 1); } private: device AssertRecorderData *ac_; }; constant constexpr int kMetalMaxNumAssertArgs = 64; constant constexpr int kMetalAssertBufferSize = sizeof(AssertRecorderData) + mtl_compute_print_msg_bytes(kMetalMaxNumAssertArgs); struct PrintMsgAllocator { atomic_int next; }; constant constexpr int kMetalPrintAssertBufferSize = 2 * 1024 * 1024; constant constexpr int kMetalPrintMsgsMaxQueueSize = kMetalPrintAssertBufferSize - sizeof(PrintMsgAllocator) - kMetalAssertBufferSize; [[maybe_unused]] device int32_t * mtl_print_alloc_buf(device PrintMsgAllocator *pa, int num_entries) { const int sz = mtl_compute_print_msg_bytes(num_entries); const int cur = atomic_fetch_add_explicit(&(pa->next), sz, metal::memory_order_relaxed); if (cur + sz >= kMetalPrintMsgsMaxQueueSize) { return (device int32_t *)0; } device byte *data_begin = reinterpret_cast<device byte *>(pa + 1); device int32_t *ptr = reinterpret_cast<device int32_t *>(data_begin + cur); *ptr = num_entries; return (ptr + 1); }

void mtl_k0003_substep_c4_0_0_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  const int tmp3 = linear_loop_idx_;
  constexpr int32_t tmp6725 = 7;
  const int32_t tmp6726 = (tmp3 >> tmp6725);
  constexpr float tmp14 = 0.0;
  S0 tmp5531(root_addr);
  constexpr int32_t tmp7429 = 0;
  S0_ch tmp5533 = tmp5531.children(tmp7429);
  S21 tmp5534 = tmp5533.get6(runtime_, mem_alloc_);
  constexpr int32_t tmp8177 = 127;
  const int32_t tmp8160 = (tmp6726 & tmp8177);
  const int32_t tmp8162 = (tmp3 & tmp8177);
  const int32_t tmp8184 = (tmp8160 << tmp6725);
  const int32_t tmp7436 = (tmp8162 + tmp8184);
  S21_ch tmp5538 = tmp5534.children(tmp7436);
  device float* tmp5539 = tmp5538.get0(runtime_, mem_alloc_).val;
  *tmp5539 = tmp14;
  device float* tmp5551 = tmp5538.get1(runtime_, mem_alloc_).val;
  *tmp5551 = tmp14;
  S24 tmp5558 = tmp5533.get7(runtime_, mem_alloc_);
  S24_ch tmp5562 = tmp5558.children(tmp7436);
  device float* tmp5563 = tmp5562.get0(runtime_, mem_alloc_).val;
  *tmp5563 = tmp14;
}

void mtl_k0003_substep_c4_0_1_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  constexpr int32_t tmp8185 = 7;
  constexpr int32_t tmp7103 = 127;
  constexpr float tmp414 = 2.0;
  constexpr int32_t tmp6887 = 16383;
  constexpr float tmp367 = 0.0078125;
  constexpr float tmp357 = 1.5258789e-05;
  constexpr float tmp352 = -0.0001;
  constexpr float tmp164 = 1e-05;
  constexpr int32_t tmp135 = 0;
  constexpr float tmp131 = 277.77777;
  constexpr float tmp129 = 416.66666;
  constexpr float tmp118 = 10.0;
  constexpr float tmp77 = 0.0001;
  constexpr float tmp64 = 0.75;
  constexpr float tmp55 = 1.5;
  constexpr float tmp42 = 128.0;
  constexpr float tmp39 = 0.3;
  constexpr float tmp38 = 0.5;
  constexpr float tmp37 = 1.0;
  constexpr float tmp36 = 0.975;
  constexpr float tmp35 = 1.0045;
  constexpr float tmp34 = 0.0;
  constexpr int32_t tmp33 = 2;
  constexpr int32_t tmp32 = 1;
  const int tmp22 = linear_loop_idx_;
  const int32_t tmp6760 = (tmp22 & tmp6887);
  constexpr int32_t tmp28 = 9000;
  const int32_t tmp29 = -(tmp6760 < tmp28);
  if (tmp29) {
    S0 tmp5566(root_addr);
    S0_ch tmp5568 = tmp5566.children(tmp135);
    S1 tmp5569 = tmp5568.get0(runtime_, mem_alloc_);
    S1_ch tmp5572 = tmp5569.children(tmp6760);
    device float* tmp5573 = tmp5572.get0(runtime_, mem_alloc_).val;
    const auto tmp41 = *tmp5573;
    const float tmp43 = (tmp41 * tmp42);
    const float tmp44 = (tmp43 - tmp38);
    const int32_t tmp45 = static_cast<int32_t>(tmp44);
    device float* tmp5583 = tmp5572.get1(runtime_, mem_alloc_).val;
    const auto tmp47 = *tmp5583;
    const float tmp48 = (tmp47 * tmp42);
    const float tmp49 = (tmp48 - tmp38);
    const int32_t tmp50 = static_cast<int32_t>(tmp49);
    const float tmp51 = static_cast<float>(tmp45);
    const float tmp52 = (tmp43 - tmp51);
    const float tmp53 = static_cast<float>(tmp50);
    const float tmp54 = (tmp48 - tmp53);
    const float tmp56 = (tmp55 - tmp52);
    const float tmp57 = (tmp56 * tmp56);
    const float tmp58 = (tmp57 * tmp38);
    const float tmp59 = (tmp55 - tmp54);
    const float tmp60 = (tmp59 * tmp59);
    const float tmp61 = (tmp60 * tmp38);
    const float tmp62 = (tmp52 - tmp37);
    const float tmp63 = (tmp62 * tmp62);
    const float tmp65 = (tmp64 - tmp63);
    const float tmp66 = (tmp54 - tmp37);
    const float tmp67 = (tmp66 * tmp66);
    const float tmp68 = (tmp64 - tmp67);
    const float tmp69 = (tmp52 - tmp38);
    const float tmp70 = (tmp69 * tmp69);
    const float tmp71 = (tmp70 * tmp38);
    const float tmp72 = (tmp54 - tmp38);
    const float tmp73 = (tmp72 * tmp72);
    const float tmp74 = (tmp73 * tmp38);
    S7 tmp5589 = tmp5568.get2(runtime_, mem_alloc_);
    S7_ch tmp5592 = tmp5589.children(tmp6760);
    device float* tmp5593 = tmp5592.get0(runtime_, mem_alloc_).val;
    const auto tmp76 = *tmp5593;
    const float tmp78 = (tmp76 * tmp77);
    const float tmp79 = (tmp78 + tmp37);
    S12 tmp5599 = tmp5568.get3(runtime_, mem_alloc_);
    S12_ch tmp5602 = tmp5599.children(tmp6760);
    device float* tmp5603 = tmp5602.get0(runtime_, mem_alloc_).val;
    const auto tmp81 = *tmp5603;
    const float tmp82 = (tmp79 * tmp81);
    device float* tmp5613 = tmp5592.get1(runtime_, mem_alloc_).val;
    const auto tmp84 = *tmp5613;
    const float tmp85 = (tmp84 * tmp77);
    device float* tmp5623 = tmp5602.get2(runtime_, mem_alloc_).val;
    const auto tmp87 = *tmp5623;
    const float tmp88 = (tmp85 * tmp87);
    const float tmp89 = (tmp82 + tmp88);
    device float* tmp5633 = tmp5602.get1(runtime_, mem_alloc_).val;
    const auto tmp91 = *tmp5633;
    const float tmp92 = (tmp79 * tmp91);
    device float* tmp5643 = tmp5602.get3(runtime_, mem_alloc_).val;
    const auto tmp94 = *tmp5643;
    const float tmp95 = (tmp85 * tmp94);
    const float tmp96 = (tmp92 + tmp95);
    device float* tmp5653 = tmp5592.get2(runtime_, mem_alloc_).val;
    const auto tmp98 = *tmp5653;
    const float tmp99 = (tmp98 * tmp77);
    const float tmp100 = (tmp99 * tmp81);
    device float* tmp5663 = tmp5592.get3(runtime_, mem_alloc_).val;
    const auto tmp102 = *tmp5663;
    const float tmp103 = (tmp102 * tmp77);
    const float tmp104 = (tmp103 + tmp37);
    const float tmp105 = (tmp104 * tmp87);
    const float tmp106 = (tmp100 + tmp105);
    const float tmp107 = (tmp99 * tmp91);
    const float tmp108 = (tmp104 * tmp94);
    const float tmp109 = (tmp107 + tmp108);
    *tmp5603 = tmp89;
    *tmp5633 = tmp96;
    *tmp5623 = tmp106;
    *tmp5643 = tmp109;
    float tmp114(0);
    S19 tmp5709 = tmp5568.get5(runtime_, mem_alloc_);
    S19_ch tmp5712 = tmp5709.children(tmp6760);
    device float* tmp5713 = tmp5712.get0(runtime_, mem_alloc_).val;
    const auto tmp116 = *tmp5713;
    const float tmp117 = (tmp37 - tmp116);
    const float tmp119 = (tmp117 * tmp118);
    const float tmp120 = exp(tmp119);
    tmp114 = tmp120;
    S17 tmp5719 = tmp5568.get4(runtime_, mem_alloc_);
    S17_ch tmp5722 = tmp5719.children(tmp6760);
    device int32_t* tmp5723 = tmp5722.get0(runtime_, mem_alloc_).val;
    const auto tmp123 = *tmp5723;
    const int32_t tmp124 = -(tmp123 == tmp32);
    const int32_t tmp125 = (tmp124 & tmp32);
    if (tmp125) {
      tmp114 = tmp39;
    } else {
    }
    const float tmp128(tmp114);
    const float tmp130 = (tmp128 * tmp129);
    const float tmp132 = (tmp128 * tmp131);
    float tmp133(0);
    tmp133 = tmp130;
    const int32_t tmp136 = -(tmp123 == tmp135);
    const int32_t tmp137 = (tmp136 & tmp32);
    if (tmp137) {
      tmp133 = tmp34;
    } else {
    }
    const float tmp140 = (tmp89 + tmp109);
    const float tmp141 = (tmp106 - tmp96);
    const float tmp142 = (tmp140 * tmp140);
    const float tmp143 = (tmp141 * tmp141);
    const float tmp144 = (tmp142 + tmp143);
    const float tmp145 = sqrt(tmp144);
    const float tmp146 = (tmp37 / tmp145);
    const float tmp147 = (tmp140 * tmp146);
    const float tmp148 = (tmp141 * tmp146);
    const float tmp149 = -(tmp148);
    const float tmp150 = (tmp147 * tmp89);
    const float tmp151 = (tmp148 * tmp106);
    const float tmp152 = (tmp150 + tmp151);
    const float tmp153 = (tmp147 * tmp96);
    const float tmp154 = (tmp148 * tmp109);
    const float tmp155 = (tmp153 + tmp154);
    const float tmp156 = (tmp149 * tmp96);
    const float tmp157 = (tmp147 * tmp109);
    const float tmp158 = (tmp156 + tmp157);
    float tmp159(0);
    float tmp160(0);
    float tmp161(0);
    float tmp162(0);
    const float tmp163 = abs(tmp155);
    const int32_t tmp165 = -(tmp163 < tmp164);
    const int32_t tmp166 = (tmp165 & tmp32);
    if (tmp166) {
      tmp159 = tmp37;
      tmp161 = tmp152;
      tmp162 = tmp158;
    } else {
      const float tmp171 = (tmp152 - tmp158);
      const float tmp172 = (tmp171 * tmp38);
      const float tmp173 = (tmp172 * tmp172);
      const float tmp174 = (tmp155 * tmp155);
      const float tmp175 = (tmp173 + tmp174);
      const float tmp176 = sqrt(tmp175);
      float tmp177(0);
      const int32_t tmp178 = -(tmp172 > tmp34);
      const int32_t tmp179 = (tmp178 & tmp32);
      if (tmp179) {
        const float tmp181 = (tmp172 + tmp176);
        const float tmp182 = (tmp155 / tmp181);
        tmp177 = tmp182;
      } else {
        const float tmp184 = (tmp172 - tmp176);
        const float tmp185 = (tmp155 / tmp184);
        tmp177 = tmp185;
      }
      const float tmp187(tmp177);
      const float tmp188 = (tmp187 * tmp187);
      const float tmp189 = (tmp188 + tmp37);
      const float tmp190 = sqrt(tmp189);
      const float tmp191 = (tmp37 / tmp190);
      tmp159 = tmp191;
      const float tmp193 = -(tmp187);
      const float tmp194 = (tmp193 * tmp191);
      tmp160 = tmp194;
      const float tmp196 = (tmp191 * tmp191);
      const float tmp197 = (tmp196 * tmp152);
      const float tmp198 = (tmp191 + tmp191);
      const float tmp199 = (tmp198 * tmp194);
      const float tmp200 = (tmp199 * tmp155);
      const float tmp201 = (tmp197 - tmp200);
      const float tmp202 = (tmp194 * tmp194);
      const float tmp203 = (tmp202 * tmp158);
      const float tmp204 = (tmp201 + tmp203);
      tmp161 = tmp204;
      const float tmp206 = (tmp202 * tmp152);
      const float tmp207 = (tmp206 + tmp200);
      const float tmp208 = (tmp196 * tmp158);
      const float tmp209 = (tmp207 + tmp208);
      tmp162 = tmp209;
    }
    float tmp211(0);
    float tmp212(0);
    float tmp213(0);
    float tmp214(0);
    const float tmp215(tmp161);
    const float tmp216(tmp162);
    const int32_t tmp217 = -(tmp215 < tmp216);
    const int32_t tmp218 = (tmp217 & tmp32);
    if (tmp218) {
      const float tmp220(tmp161);
      const float tmp221(tmp162);
      tmp161 = tmp221;
      tmp162 = tmp220;
      const float tmp224(tmp160);
      const float tmp225 = -(tmp224);
      const float tmp226(tmp159);
      const float tmp227 = -(tmp226);
      tmp211 = tmp225;
      tmp212 = tmp226;
      tmp213 = tmp227;
      tmp214 = tmp225;
    } else {
      const float tmp232(tmp159);
      const float tmp233(tmp160);
      const float tmp234 = -(tmp233);
      tmp211 = tmp232;
      tmp212 = tmp233;
      tmp213 = tmp234;
      tmp214 = tmp232;
    }
    const float tmp239(tmp211);
    const float tmp240 = (tmp147 * tmp239);
    const float tmp241(tmp213);
    const float tmp242 = (tmp149 * tmp241);
    const float tmp243 = (tmp240 + tmp242);
    const float tmp244(tmp212);
    const float tmp245 = (tmp147 * tmp244);
    const float tmp246(tmp214);
    const float tmp247 = (tmp149 * tmp246);
    const float tmp248 = (tmp245 + tmp247);
    const float tmp249 = (tmp148 * tmp239);
    const float tmp250 = (tmp147 * tmp241);
    const float tmp251 = (tmp249 + tmp250);
    const float tmp252 = (tmp148 * tmp244);
    const float tmp253 = (tmp147 * tmp246);
    const float tmp254 = (tmp252 + tmp253);
    const float tmp255(tmp161);
    const float tmp256(tmp162);
    float tmp257(0);
    tmp257 = tmp255;
    const int32_t tmp259 = -(tmp123 == tmp33);
    const int32_t tmp260 = (tmp259 & tmp32);
    if (tmp260) {
      const float tmp262 =  max(tmp255, tmp36);
      const float tmp263 =  min(tmp262, tmp35);
      tmp257 = tmp263;
    } else {
    }
    const float tmp265(tmp257);
    const float tmp266 = (tmp255 / tmp265);
    const float tmp267 = (tmp116 * tmp266);
    float tmp268(0);
    tmp268 = tmp256;
    if (tmp260) {
      const float tmp271 =  max(tmp256, tmp36);
      const float tmp272 =  min(tmp271, tmp35);
      tmp268 = tmp272;
    } else {
    }
    const float tmp274(tmp268);
    const float tmp275 = (tmp256 / tmp274);
    const float tmp276 = (tmp267 * tmp275);
    *tmp5713 = tmp276;
    const float tmp278 = (tmp265 * tmp274);
    if (tmp137) {
      const float tmp280 = sqrt(tmp278);
      *tmp5603 = tmp280;
      *tmp5633 = tmp34;
      *tmp5623 = tmp34;
      *tmp5643 = tmp280;
    } else {
      const auto tmp285 = *tmp5723;
      const int32_t tmp286 = -(tmp285 == tmp33);
      const int32_t tmp287 = (tmp286 & tmp32);
      if (tmp287) {
        const float tmp289 = (tmp243 * tmp265);
        const float tmp290 = (tmp248 * tmp274);
        const float tmp291 = (tmp251 * tmp265);
        const float tmp292 = (tmp254 * tmp274);
        const float tmp293 = (tmp289 * tmp239);
        const float tmp294 = (tmp290 * tmp244);
        const float tmp295 = (tmp293 + tmp294);
        const float tmp296 = (tmp289 * tmp241);
        const float tmp297 = (tmp290 * tmp246);
        const float tmp298 = (tmp296 + tmp297);
        const float tmp299 = (tmp291 * tmp239);
        const float tmp300 = (tmp292 * tmp244);
        const float tmp301 = (tmp299 + tmp300);
        const float tmp302 = (tmp291 * tmp241);
        const float tmp303 = (tmp292 * tmp246);
        const float tmp304 = (tmp302 + tmp303);
        *tmp5603 = tmp295;
        *tmp5633 = tmp298;
        *tmp5623 = tmp301;
        *tmp5643 = tmp304;
      } else {
      }
    }
    const float tmp309 = (tmp243 * tmp239);
    const float tmp310 = (tmp248 * tmp244);
    const float tmp311 = (tmp309 + tmp310);
    const float tmp312 = (tmp243 * tmp241);
    const float tmp313 = (tmp248 * tmp246);
    const float tmp314 = (tmp312 + tmp313);
    const float tmp315 = (tmp251 * tmp239);
    const float tmp316 = (tmp254 * tmp244);
    const float tmp317 = (tmp315 + tmp316);
    const float tmp318 = (tmp251 * tmp241);
    const float tmp319 = (tmp254 * tmp246);
    const float tmp320 = (tmp318 + tmp319);
    const auto tmp321 = *tmp5603;
    const auto tmp322 = *tmp5623;
    const auto tmp323 = *tmp5633;
    const auto tmp324 = *tmp5643;
    const float tmp325(tmp133);
    const float tmp326 = (tmp325 + tmp325);
    const float tmp327 = (tmp321 - tmp311);
    const float tmp328 = (tmp326 * tmp327);
    const float tmp329 = (tmp328 * tmp321);
    const float tmp330 = (tmp323 - tmp314);
    const float tmp331 = (tmp326 * tmp330);
    const float tmp332 = (tmp331 * tmp323);
    const float tmp333 = (tmp329 + tmp332);
    const float tmp334 = (tmp328 * tmp322);
    const float tmp335 = (tmp331 * tmp324);
    const float tmp336 = (tmp334 + tmp335);
    const float tmp337 = (tmp322 - tmp317);
    const float tmp338 = (tmp326 * tmp337);
    const float tmp339 = (tmp338 * tmp321);
    const float tmp340 = (tmp324 - tmp320);
    const float tmp341 = (tmp326 * tmp340);
    const float tmp342 = (tmp341 * tmp323);
    const float tmp343 = (tmp339 + tmp342);
    const float tmp344 = (tmp338 * tmp322);
    const float tmp345 = (tmp341 * tmp324);
    const float tmp346 = (tmp344 + tmp345);
    const float tmp347 = (tmp132 * tmp278);
    const float tmp348 = (tmp278 - tmp37);
    const float tmp349 = (tmp347 * tmp348);
    const float tmp350 = (tmp333 + tmp349);
    const float tmp351 = (tmp346 + tmp349);
    const float tmp353 = (tmp350 * tmp352);
    const float tmp354 = (tmp336 * tmp352);
    const float tmp355 = (tmp343 * tmp352);
    const float tmp356 = (tmp351 * tmp352);
    const float tmp358 = (tmp76 * tmp357);
    const float tmp359 = (tmp353 + tmp358);
    const float tmp360 = (tmp84 * tmp357);
    const float tmp361 = (tmp354 + tmp360);
    const float tmp362 = (tmp98 * tmp357);
    const float tmp363 = (tmp355 + tmp362);
    const float tmp364 = (tmp102 * tmp357);
    const float tmp365 = (tmp356 + tmp364);
    const float tmp366 = (tmp34 - tmp52);
    const float tmp368 = (tmp366 * tmp367);
    const float tmp369 = (tmp34 - tmp54);
    const float tmp370 = (tmp369 * tmp367);
    const float tmp371 = (tmp58 * tmp61);
    const float tmp372 = (tmp359 * tmp368);
    const float tmp373 = (tmp361 * tmp370);
    const float tmp374 = (tmp372 + tmp373);
    const float tmp375 = (tmp363 * tmp368);
    const float tmp376 = (tmp365 * tmp370);
    const float tmp377 = (tmp375 + tmp376);
    S4 tmp5869 = tmp5568.get1(runtime_, mem_alloc_);
    S4_ch tmp5872 = tmp5869.children(tmp6760);
    device float* tmp5873 = tmp5872.get0(runtime_, mem_alloc_).val;
    const auto tmp379 = *tmp5873;
    const float tmp380 = (tmp379 * tmp357);
    const float tmp381 = (tmp380 + tmp374);
    const float tmp382 = (tmp371 * tmp381);
    device float* tmp5883 = tmp5872.get1(runtime_, mem_alloc_).val;
    const auto tmp384 = *tmp5883;
    const float tmp385 = (tmp384 * tmp357);
    const float tmp386 = (tmp385 + tmp377);
    const float tmp387 = (tmp371 * tmp386);
    S21 tmp5890 = tmp5568.get6(runtime_, mem_alloc_);
    const int32_t tmp6892 = (tmp45 & tmp7103);
    const int32_t tmp6896 = (tmp50 & tmp7103);
    const int32_t tmp8186 = (tmp6892 << tmp8185);
    const int32_t tmp7575 = (tmp6896 + tmp8186);
    S21_ch tmp5894 = tmp5890.children(tmp7575);
    device float* tmp5895 = tmp5894.get0(runtime_, mem_alloc_).val;
    const float tmp389 = fatomic_fetch_add(tmp5895, tmp382);
    device float* tmp5907 = tmp5894.get1(runtime_, mem_alloc_).val;
    const float tmp391 = fatomic_fetch_add(tmp5907, tmp387);
    const float tmp392 = (tmp371 * tmp357);
    S24 tmp5914 = tmp5568.get7(runtime_, mem_alloc_);
    S24_ch tmp5918 = tmp5914.children(tmp7575);
    device float* tmp5919 = tmp5918.get0(runtime_, mem_alloc_).val;
    const float tmp394 = fatomic_fetch_add(tmp5919, tmp392);
    const float tmp395 = (tmp37 - tmp54);
    const float tmp396 = (tmp395 * tmp367);
    const float tmp397 = (tmp58 * tmp68);
    const float tmp398 = (tmp361 * tmp396);
    const float tmp399 = (tmp372 + tmp398);
    const float tmp400 = (tmp365 * tmp396);
    const float tmp401 = (tmp375 + tmp400);
    const float tmp402 = (tmp380 + tmp399);
    const float tmp403 = (tmp397 * tmp402);
    const float tmp404 = (tmp385 + tmp401);
    const float tmp405 = (tmp397 * tmp404);
    const int32_t tmp406 = (tmp50 + tmp32);
    const int32_t tmp6920 = (tmp406 & tmp7103);
    const int32_t tmp7599 = (tmp6920 + tmp8186);
    S21_ch tmp5930 = tmp5890.children(tmp7599);
    device float* tmp5931 = tmp5930.get0(runtime_, mem_alloc_).val;
    const float tmp408 = fatomic_fetch_add(tmp5931, tmp403);
    device float* tmp5943 = tmp5930.get1(runtime_, mem_alloc_).val;
    const float tmp410 = fatomic_fetch_add(tmp5943, tmp405);
    const float tmp411 = (tmp397 * tmp357);
    S24_ch tmp5954 = tmp5914.children(tmp7599);
    device float* tmp5955 = tmp5954.get0(runtime_, mem_alloc_).val;
    const float tmp413 = fatomic_fetch_add(tmp5955, tmp411);
    const float tmp415 = (tmp414 - tmp54);
    const float tmp416 = (tmp415 * tmp367);
    const float tmp417 = (tmp58 * tmp74);
    const float tmp418 = (tmp361 * tmp416);
    const float tmp419 = (tmp372 + tmp418);
    const float tmp420 = (tmp365 * tmp416);
    const float tmp421 = (tmp375 + tmp420);
    const float tmp422 = (tmp380 + tmp419);
    const float tmp423 = (tmp417 * tmp422);
    const float tmp424 = (tmp385 + tmp421);
    const float tmp425 = (tmp417 * tmp424);
    const int32_t tmp426 = (tmp50 + tmp33);
    const int32_t tmp6944 = (tmp426 & tmp7103);
    const int32_t tmp7623 = (tmp6944 + tmp8186);
    S21_ch tmp5966 = tmp5890.children(tmp7623);
    device float* tmp5967 = tmp5966.get0(runtime_, mem_alloc_).val;
    const float tmp428 = fatomic_fetch_add(tmp5967, tmp423);
    device float* tmp5979 = tmp5966.get1(runtime_, mem_alloc_).val;
    const float tmp430 = fatomic_fetch_add(tmp5979, tmp425);
    const float tmp431 = (tmp417 * tmp357);
    S24_ch tmp5990 = tmp5914.children(tmp7623);
    device float* tmp5991 = tmp5990.get0(runtime_, mem_alloc_).val;
    const float tmp433 = fatomic_fetch_add(tmp5991, tmp431);
    const float tmp434 = (tmp37 - tmp52);
    const float tmp435 = (tmp434 * tmp367);
    const float tmp436 = (tmp65 * tmp61);
    const float tmp437 = (tmp359 * tmp435);
    const float tmp438 = (tmp437 + tmp373);
    const float tmp439 = (tmp363 * tmp435);
    const float tmp440 = (tmp439 + tmp376);
    const float tmp441 = (tmp380 + tmp438);
    const float tmp442 = (tmp436 * tmp441);
    const float tmp443 = (tmp385 + tmp440);
    const float tmp444 = (tmp436 * tmp443);
    const int32_t tmp445 = (tmp45 + tmp32);
    const int32_t tmp6964 = (tmp445 & tmp7103);
    const int32_t tmp8188 = (tmp6964 << tmp8185);
    const int32_t tmp7647 = (tmp6896 + tmp8188);
    S21_ch tmp6002 = tmp5890.children(tmp7647);
    device float* tmp6003 = tmp6002.get0(runtime_, mem_alloc_).val;
    const float tmp447 = fatomic_fetch_add(tmp6003, tmp442);
    device float* tmp6015 = tmp6002.get1(runtime_, mem_alloc_).val;
    const float tmp449 = fatomic_fetch_add(tmp6015, tmp444);
    const float tmp450 = (tmp436 * tmp357);
    S24_ch tmp6026 = tmp5914.children(tmp7647);
    device float* tmp6027 = tmp6026.get0(runtime_, mem_alloc_).val;
    const float tmp452 = fatomic_fetch_add(tmp6027, tmp450);
    const float tmp453 = (tmp65 * tmp68);
    const float tmp454 = (tmp437 + tmp398);
    const float tmp455 = (tmp439 + tmp400);
    const float tmp456 = (tmp380 + tmp454);
    const float tmp457 = (tmp453 * tmp456);
    const float tmp458 = (tmp385 + tmp455);
    const float tmp459 = (tmp453 * tmp458);
    const int32_t tmp7671 = (tmp6920 + tmp8188);
    S21_ch tmp6038 = tmp5890.children(tmp7671);
    device float* tmp6039 = tmp6038.get0(runtime_, mem_alloc_).val;
    const float tmp461 = fatomic_fetch_add(tmp6039, tmp457);
    device float* tmp6051 = tmp6038.get1(runtime_, mem_alloc_).val;
    const float tmp463 = fatomic_fetch_add(tmp6051, tmp459);
    const float tmp464 = (tmp453 * tmp357);
    S24_ch tmp6062 = tmp5914.children(tmp7671);
    device float* tmp6063 = tmp6062.get0(runtime_, mem_alloc_).val;
    const float tmp466 = fatomic_fetch_add(tmp6063, tmp464);
    const float tmp467 = (tmp65 * tmp74);
    const float tmp468 = (tmp437 + tmp418);
    const float tmp469 = (tmp439 + tmp420);
    const float tmp470 = (tmp380 + tmp468);
    const float tmp471 = (tmp467 * tmp470);
    const float tmp472 = (tmp385 + tmp469);
    const float tmp473 = (tmp467 * tmp472);
    const int32_t tmp7695 = (tmp6944 + tmp8188);
    S21_ch tmp6074 = tmp5890.children(tmp7695);
    device float* tmp6075 = tmp6074.get0(runtime_, mem_alloc_).val;
    const float tmp475 = fatomic_fetch_add(tmp6075, tmp471);
    device float* tmp6087 = tmp6074.get1(runtime_, mem_alloc_).val;
    const float tmp477 = fatomic_fetch_add(tmp6087, tmp473);
    const float tmp478 = (tmp467 * tmp357);
    S24_ch tmp6098 = tmp5914.children(tmp7695);
    device float* tmp6099 = tmp6098.get0(runtime_, mem_alloc_).val;
    const float tmp480 = fatomic_fetch_add(tmp6099, tmp478);
    const float tmp481 = (tmp414 - tmp52);
    const float tmp482 = (tmp481 * tmp367);
    const float tmp483 = (tmp71 * tmp61);
    const float tmp484 = (tmp359 * tmp482);
    const float tmp485 = (tmp484 + tmp373);
    const float tmp486 = (tmp363 * tmp482);
    const float tmp487 = (tmp486 + tmp376);
    const float tmp488 = (tmp380 + tmp485);
    const float tmp489 = (tmp483 * tmp488);
    const float tmp490 = (tmp385 + tmp487);
    const float tmp491 = (tmp483 * tmp490);
    const int32_t tmp492 = (tmp45 + tmp33);
    const int32_t tmp7036 = (tmp492 & tmp7103);
    const int32_t tmp8190 = (tmp7036 << tmp8185);
    const int32_t tmp7719 = (tmp6896 + tmp8190);
    S21_ch tmp6110 = tmp5890.children(tmp7719);
    device float* tmp6111 = tmp6110.get0(runtime_, mem_alloc_).val;
    const float tmp494 = fatomic_fetch_add(tmp6111, tmp489);
    device float* tmp6123 = tmp6110.get1(runtime_, mem_alloc_).val;
    const float tmp496 = fatomic_fetch_add(tmp6123, tmp491);
    const float tmp497 = (tmp483 * tmp357);
    S24_ch tmp6134 = tmp5914.children(tmp7719);
    device float* tmp6135 = tmp6134.get0(runtime_, mem_alloc_).val;
    const float tmp499 = fatomic_fetch_add(tmp6135, tmp497);
    const float tmp500 = (tmp71 * tmp68);
    const float tmp501 = (tmp484 + tmp398);
    const float tmp502 = (tmp486 + tmp400);
    const float tmp503 = (tmp380 + tmp501);
    const float tmp504 = (tmp500 * tmp503);
    const float tmp505 = (tmp385 + tmp502);
    const float tmp506 = (tmp500 * tmp505);
    const int32_t tmp7743 = (tmp6920 + tmp8190);
    S21_ch tmp6146 = tmp5890.children(tmp7743);
    device float* tmp6147 = tmp6146.get0(runtime_, mem_alloc_).val;
    const float tmp508 = fatomic_fetch_add(tmp6147, tmp504);
    device float* tmp6159 = tmp6146.get1(runtime_, mem_alloc_).val;
    const float tmp510 = fatomic_fetch_add(tmp6159, tmp506);
    const float tmp511 = (tmp500 * tmp357);
    S24_ch tmp6170 = tmp5914.children(tmp7743);
    device float* tmp6171 = tmp6170.get0(runtime_, mem_alloc_).val;
    const float tmp513 = fatomic_fetch_add(tmp6171, tmp511);
    const float tmp514 = (tmp71 * tmp74);
    const float tmp515 = (tmp484 + tmp418);
    const float tmp516 = (tmp486 + tmp420);
    const float tmp517 = (tmp380 + tmp515);
    const float tmp518 = (tmp514 * tmp517);
    const float tmp519 = (tmp385 + tmp516);
    const float tmp520 = (tmp514 * tmp519);
    const int32_t tmp7767 = (tmp6944 + tmp8190);
    S21_ch tmp6182 = tmp5890.children(tmp7767);
    device float* tmp6183 = tmp6182.get0(runtime_, mem_alloc_).val;
    const float tmp522 = fatomic_fetch_add(tmp6183, tmp518);
    device float* tmp6195 = tmp6182.get1(runtime_, mem_alloc_).val;
    const float tmp524 = fatomic_fetch_add(tmp6195, tmp520);
    const float tmp525 = (tmp514 * tmp357);
    S24_ch tmp6206 = tmp5914.children(tmp7767);
    device float* tmp6207 = tmp6206.get0(runtime_, mem_alloc_).val;
    const float tmp527 = fatomic_fetch_add(tmp6207, tmp525);
  } else {
  }
}

void mtl_k0003_substep_c4_0_2_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  constexpr int32_t tmp7231 = 127;
  const int tmp531 = linear_loop_idx_;
  constexpr int32_t tmp7105 = 7;
  const int32_t tmp7106 = (tmp531 >> tmp7105);
  const int32_t tmp7108 = (tmp7106 & tmp7231);
  const int32_t tmp7112 = (tmp531 & tmp7231);
  constexpr float tmp541 = 0.0;
  constexpr int32_t tmp542 = 1;
  constexpr int32_t tmp543 = 125;
  constexpr int32_t tmp544 = 3;
  constexpr float tmp545 = -0.005;
  constexpr float tmp546 = 1.0;
  S0 tmp6211(root_addr);
  constexpr int32_t tmp7829 = 0;
  S0_ch tmp6213 = tmp6211.children(tmp7829);
  S24 tmp6214 = tmp6213.get7(runtime_, mem_alloc_);
  const int32_t tmp8192 = (tmp7108 << tmp7105);
  const int32_t tmp7836 = (tmp7112 + tmp8192);
  S24_ch tmp6218 = tmp6214.children(tmp7836);
  device float* tmp6219 = tmp6218.get0(runtime_, mem_alloc_).val;
  const auto tmp548 = *tmp6219;
  const int32_t tmp549 = -(tmp548 > tmp541);
  const int32_t tmp550 = (tmp549 & tmp542);
  if (tmp550) {
    const auto tmp552 = *tmp6219;
    const float tmp553 = (tmp546 / tmp552);
    S21 tmp6238 = tmp6213.get6(runtime_, mem_alloc_);
    S21_ch tmp6242 = tmp6238.children(tmp7836);
    device float* tmp6243 = tmp6242.get0(runtime_, mem_alloc_).val;
    const auto tmp555 = *tmp6243;
    const float tmp556 = (tmp553 * tmp555);
    device float* tmp6255 = tmp6242.get1(runtime_, mem_alloc_).val;
    const auto tmp558 = *tmp6255;
    const float tmp559 = (tmp553 * tmp558);
    *tmp6243 = tmp556;
    *tmp6255 = tmp559;
    const auto tmp562 = *tmp6255;
    const float tmp563 = (tmp562 + tmp545);
    *tmp6255 = tmp563;
    const int32_t tmp565 = -(tmp7108 < tmp544);
    const int32_t tmp566 = (tmp565 & tmp542);
    const int32_t tmp567 = -(tmp556 < tmp541);
    const int32_t tmp568 = (tmp567 & tmp542);
    const int32_t tmp569 = (tmp566 & tmp568);
    if (tmp569) {
      *tmp6243 = tmp541;
    } else {
    }
    const int32_t tmp572 = -(tmp7108 > tmp543);
    const int32_t tmp573 = (tmp572 & tmp542);
    const auto tmp574 = *tmp6243;
    const int32_t tmp575 = -(tmp574 > tmp541);
    const int32_t tmp576 = (tmp575 & tmp542);
    const int32_t tmp577 = (tmp573 & tmp576);
    if (tmp577) {
      *tmp6243 = tmp541;
    } else {
    }
    const int32_t tmp580 = -(tmp7112 < tmp544);
    const int32_t tmp581 = (tmp580 & tmp542);
    const auto tmp582 = *tmp6255;
    const int32_t tmp583 = -(tmp582 < tmp541);
    const int32_t tmp584 = (tmp583 & tmp542);
    const int32_t tmp585 = (tmp581 & tmp584);
    if (tmp585) {
      *tmp6255 = tmp541;
    } else {
    }
    const int32_t tmp588 = -(tmp7112 > tmp543);
    const int32_t tmp589 = (tmp588 & tmp542);
    const auto tmp590 = *tmp6255;
    const int32_t tmp591 = -(tmp590 > tmp541);
    const int32_t tmp592 = (tmp591 & tmp542);
    const int32_t tmp593 = (tmp589 & tmp592);
    if (tmp593) {
      *tmp6255 = tmp541;
    } else {
    }
  } else {
  }
}

void mtl_k0003_substep_c4_0_3_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  constexpr int32_t tmp8193 = 7;
  constexpr int32_t tmp7949 = 0;
  constexpr int32_t tmp7427 = 16383;
  constexpr float tmp990 = 0.0001;
  constexpr int32_t tmp7387 = 127;
  constexpr int32_t tmp728 = 2;
  constexpr float tmp726 = 2.0;
  constexpr int32_t tmp690 = 1;
  constexpr float tmp671 = 512.0;
  constexpr float tmp651 = 0.0;
  constexpr float tmp634 = 0.75;
  constexpr float tmp631 = 1.0;
  constexpr float tmp624 = 1.5;
  constexpr float tmp612 = 0.5;
  constexpr float tmp610 = 128.0;
  const int tmp598 = linear_loop_idx_;
  const int32_t tmp7236 = (tmp598 & tmp7427);
  constexpr int32_t tmp604 = 9000;
  const int32_t tmp605 = -(tmp7236 < tmp604);
  if (tmp605) {
    S0 tmp6390(root_addr);
    S0_ch tmp6392 = tmp6390.children(tmp7949);
    S1 tmp6393 = tmp6392.get0(runtime_, mem_alloc_);
    S1_ch tmp6396 = tmp6393.children(tmp7236);
    device float* tmp6397 = tmp6396.get0(runtime_, mem_alloc_).val;
    const auto tmp609 = *tmp6397;
    const float tmp611 = (tmp609 * tmp610);
    const float tmp613 = (tmp611 - tmp612);
    const int32_t tmp614 = static_cast<int32_t>(tmp613);
    device float* tmp6407 = tmp6396.get1(runtime_, mem_alloc_).val;
    const auto tmp616 = *tmp6407;
    const float tmp617 = (tmp616 * tmp610);
    const float tmp618 = (tmp617 - tmp612);
    const int32_t tmp619 = static_cast<int32_t>(tmp618);
    const float tmp620 = static_cast<float>(tmp614);
    const float tmp621 = (tmp611 - tmp620);
    const float tmp622 = static_cast<float>(tmp619);
    const float tmp623 = (tmp617 - tmp622);
    const float tmp625 = (tmp624 - tmp621);
    const float tmp626 = (tmp625 * tmp625);
    const float tmp627 = (tmp626 * tmp612);
    const float tmp628 = (tmp624 - tmp623);
    const float tmp629 = (tmp628 * tmp628);
    const float tmp630 = (tmp629 * tmp612);
    const float tmp632 = (tmp621 - tmp631);
    const float tmp633 = (tmp632 * tmp632);
    const float tmp635 = (tmp634 - tmp633);
    const float tmp636 = (tmp623 - tmp631);
    const float tmp637 = (tmp636 * tmp636);
    const float tmp638 = (tmp634 - tmp637);
    const float tmp639 = (tmp621 - tmp612);
    const float tmp640 = (tmp639 * tmp639);
    const float tmp641 = (tmp640 * tmp612);
    const float tmp642 = (tmp623 - tmp612);
    const float tmp643 = (tmp642 * tmp642);
    const float tmp644 = (tmp643 * tmp612);
    const float tmp652 = (tmp651 - tmp621);
    const float tmp653 = (tmp651 - tmp623);
    S21 tmp6414 = tmp6392.get6(runtime_, mem_alloc_);
    const int32_t tmp7248 = (tmp614 & tmp7387);
    const int32_t tmp7252 = (tmp619 & tmp7387);
    const int32_t tmp8194 = (tmp7248 << tmp8193);
    const int32_t tmp7966 = (tmp7252 + tmp8194);
    S21_ch tmp6418 = tmp6414.children(tmp7966);
    device float* tmp6419 = tmp6418.get0(runtime_, mem_alloc_).val;
    const auto tmp655 = *tmp6419;
    device float* tmp6431 = tmp6418.get1(runtime_, mem_alloc_).val;
    const auto tmp657 = *tmp6431;
    const float tmp658 = (tmp627 * tmp630);
    const float tmp659 = (tmp658 * tmp655);
    const float tmp660 = (tmp658 * tmp657);
    const float tmp667 = (tmp655 * tmp652);
    const float tmp668 = (tmp655 * tmp653);
    const float tmp669 = (tmp657 * tmp652);
    const float tmp670 = (tmp657 * tmp653);
    const float tmp672 = (tmp658 * tmp671);
    const float tmp673 = (tmp672 * tmp667);
    const float tmp674 = (tmp672 * tmp668);
    const float tmp675 = (tmp672 * tmp669);
    const float tmp676 = (tmp672 * tmp670);
    const float tmp689 = (tmp631 - tmp623);
    const int32_t tmp691 = (tmp619 + tmp690);
    const int32_t tmp7268 = (tmp691 & tmp7387);
    const int32_t tmp7982 = (tmp7268 + tmp8194);
    S21_ch tmp6442 = tmp6414.children(tmp7982);
    device float* tmp6443 = tmp6442.get0(runtime_, mem_alloc_).val;
    const auto tmp693 = *tmp6443;
    device float* tmp6455 = tmp6442.get1(runtime_, mem_alloc_).val;
    const auto tmp695 = *tmp6455;
    const float tmp696 = (tmp627 * tmp638);
    const float tmp697 = (tmp696 * tmp693);
    const float tmp698 = (tmp696 * tmp695);
    const float tmp700 = (tmp659 + tmp697);
    const float tmp703 = (tmp660 + tmp698);
    const float tmp705 = (tmp693 * tmp652);
    const float tmp706 = (tmp693 * tmp689);
    const float tmp707 = (tmp695 * tmp652);
    const float tmp708 = (tmp695 * tmp689);
    const float tmp709 = (tmp696 * tmp671);
    const float tmp710 = (tmp709 * tmp705);
    const float tmp711 = (tmp709 * tmp706);
    const float tmp712 = (tmp709 * tmp707);
    const float tmp713 = (tmp709 * tmp708);
    const float tmp715 = (tmp673 + tmp710);
    const float tmp718 = (tmp674 + tmp711);
    const float tmp721 = (tmp675 + tmp712);
    const float tmp724 = (tmp676 + tmp713);
    const float tmp727 = (tmp726 - tmp623);
    const int32_t tmp729 = (tmp619 + tmp728);
    const int32_t tmp7284 = (tmp729 & tmp7387);
    const int32_t tmp7998 = (tmp7284 + tmp8194);
    S21_ch tmp6466 = tmp6414.children(tmp7998);
    device float* tmp6467 = tmp6466.get0(runtime_, mem_alloc_).val;
    const auto tmp731 = *tmp6467;
    device float* tmp6479 = tmp6466.get1(runtime_, mem_alloc_).val;
    const auto tmp733 = *tmp6479;
    const float tmp734 = (tmp627 * tmp644);
    const float tmp735 = (tmp734 * tmp731);
    const float tmp736 = (tmp734 * tmp733);
    const float tmp738 = (tmp700 + tmp735);
    const float tmp741 = (tmp703 + tmp736);
    const float tmp743 = (tmp731 * tmp652);
    const float tmp744 = (tmp731 * tmp727);
    const float tmp745 = (tmp733 * tmp652);
    const float tmp746 = (tmp733 * tmp727);
    const float tmp747 = (tmp734 * tmp671);
    const float tmp748 = (tmp747 * tmp743);
    const float tmp749 = (tmp747 * tmp744);
    const float tmp750 = (tmp747 * tmp745);
    const float tmp751 = (tmp747 * tmp746);
    const float tmp753 = (tmp715 + tmp748);
    const float tmp756 = (tmp718 + tmp749);
    const float tmp759 = (tmp721 + tmp750);
    const float tmp762 = (tmp724 + tmp751);
    const float tmp764 = (tmp631 - tmp621);
    const int32_t tmp765 = (tmp614 + tmp690);
    const int32_t tmp7296 = (tmp765 & tmp7387);
    const int32_t tmp8196 = (tmp7296 << tmp8193);
    const int32_t tmp8014 = (tmp7252 + tmp8196);
    S21_ch tmp6490 = tmp6414.children(tmp8014);
    device float* tmp6491 = tmp6490.get0(runtime_, mem_alloc_).val;
    const auto tmp767 = *tmp6491;
    device float* tmp6503 = tmp6490.get1(runtime_, mem_alloc_).val;
    const auto tmp769 = *tmp6503;
    const float tmp770 = (tmp635 * tmp630);
    const float tmp771 = (tmp770 * tmp767);
    const float tmp772 = (tmp770 * tmp769);
    const float tmp774 = (tmp738 + tmp771);
    const float tmp777 = (tmp741 + tmp772);
    const float tmp779 = (tmp767 * tmp764);
    const float tmp780 = (tmp767 * tmp653);
    const float tmp781 = (tmp769 * tmp764);
    const float tmp782 = (tmp769 * tmp653);
    const float tmp783 = (tmp770 * tmp671);
    const float tmp784 = (tmp783 * tmp779);
    const float tmp785 = (tmp783 * tmp780);
    const float tmp786 = (tmp783 * tmp781);
    const float tmp787 = (tmp783 * tmp782);
    const float tmp789 = (tmp753 + tmp784);
    const float tmp792 = (tmp756 + tmp785);
    const float tmp795 = (tmp759 + tmp786);
    const float tmp798 = (tmp762 + tmp787);
    const int32_t tmp8030 = (tmp7268 + tmp8196);
    S21_ch tmp6514 = tmp6414.children(tmp8030);
    device float* tmp6515 = tmp6514.get0(runtime_, mem_alloc_).val;
    const auto tmp801 = *tmp6515;
    device float* tmp6527 = tmp6514.get1(runtime_, mem_alloc_).val;
    const auto tmp803 = *tmp6527;
    const float tmp804 = (tmp635 * tmp638);
    const float tmp805 = (tmp804 * tmp801);
    const float tmp806 = (tmp804 * tmp803);
    const float tmp808 = (tmp774 + tmp805);
    const float tmp811 = (tmp777 + tmp806);
    const float tmp813 = (tmp801 * tmp764);
    const float tmp814 = (tmp801 * tmp689);
    const float tmp815 = (tmp803 * tmp764);
    const float tmp816 = (tmp803 * tmp689);
    const float tmp817 = (tmp804 * tmp671);
    const float tmp818 = (tmp817 * tmp813);
    const float tmp819 = (tmp817 * tmp814);
    const float tmp820 = (tmp817 * tmp815);
    const float tmp821 = (tmp817 * tmp816);
    const float tmp823 = (tmp789 + tmp818);
    const float tmp826 = (tmp792 + tmp819);
    const float tmp829 = (tmp795 + tmp820);
    const float tmp832 = (tmp798 + tmp821);
    const int32_t tmp8046 = (tmp7284 + tmp8196);
    S21_ch tmp6538 = tmp6414.children(tmp8046);
    device float* tmp6539 = tmp6538.get0(runtime_, mem_alloc_).val;
    const auto tmp835 = *tmp6539;
    device float* tmp6551 = tmp6538.get1(runtime_, mem_alloc_).val;
    const auto tmp837 = *tmp6551;
    const float tmp838 = (tmp635 * tmp644);
    const float tmp839 = (tmp838 * tmp835);
    const float tmp840 = (tmp838 * tmp837);
    const float tmp842 = (tmp808 + tmp839);
    const float tmp845 = (tmp811 + tmp840);
    const float tmp847 = (tmp835 * tmp764);
    const float tmp848 = (tmp835 * tmp727);
    const float tmp849 = (tmp837 * tmp764);
    const float tmp850 = (tmp837 * tmp727);
    const float tmp851 = (tmp838 * tmp671);
    const float tmp852 = (tmp851 * tmp847);
    const float tmp853 = (tmp851 * tmp848);
    const float tmp854 = (tmp851 * tmp849);
    const float tmp855 = (tmp851 * tmp850);
    const float tmp857 = (tmp823 + tmp852);
    const float tmp860 = (tmp826 + tmp853);
    const float tmp863 = (tmp829 + tmp854);
    const float tmp866 = (tmp832 + tmp855);
    const float tmp868 = (tmp726 - tmp621);
    const int32_t tmp869 = (tmp614 + tmp728);
    const int32_t tmp7344 = (tmp869 & tmp7387);
    const int32_t tmp8198 = (tmp7344 << tmp8193);
    const int32_t tmp8062 = (tmp7252 + tmp8198);
    S21_ch tmp6562 = tmp6414.children(tmp8062);
    device float* tmp6563 = tmp6562.get0(runtime_, mem_alloc_).val;
    const auto tmp871 = *tmp6563;
    device float* tmp6575 = tmp6562.get1(runtime_, mem_alloc_).val;
    const auto tmp873 = *tmp6575;
    const float tmp874 = (tmp641 * tmp630);
    const float tmp875 = (tmp874 * tmp871);
    const float tmp876 = (tmp874 * tmp873);
    const float tmp878 = (tmp842 + tmp875);
    const float tmp881 = (tmp845 + tmp876);
    const float tmp883 = (tmp871 * tmp868);
    const float tmp884 = (tmp871 * tmp653);
    const float tmp885 = (tmp873 * tmp868);
    const float tmp886 = (tmp873 * tmp653);
    const float tmp887 = (tmp874 * tmp671);
    const float tmp888 = (tmp887 * tmp883);
    const float tmp889 = (tmp887 * tmp884);
    const float tmp890 = (tmp887 * tmp885);
    const float tmp891 = (tmp887 * tmp886);
    const float tmp893 = (tmp857 + tmp888);
    const float tmp896 = (tmp860 + tmp889);
    const float tmp899 = (tmp863 + tmp890);
    const float tmp902 = (tmp866 + tmp891);
    const int32_t tmp8078 = (tmp7268 + tmp8198);
    S21_ch tmp6586 = tmp6414.children(tmp8078);
    device float* tmp6587 = tmp6586.get0(runtime_, mem_alloc_).val;
    const auto tmp905 = *tmp6587;
    device float* tmp6599 = tmp6586.get1(runtime_, mem_alloc_).val;
    const auto tmp907 = *tmp6599;
    const float tmp908 = (tmp641 * tmp638);
    const float tmp909 = (tmp908 * tmp905);
    const float tmp910 = (tmp908 * tmp907);
    const float tmp912 = (tmp878 + tmp909);
    const float tmp915 = (tmp881 + tmp910);
    const float tmp917 = (tmp905 * tmp868);
    const float tmp918 = (tmp905 * tmp689);
    const float tmp919 = (tmp907 * tmp868);
    const float tmp920 = (tmp907 * tmp689);
    const float tmp921 = (tmp908 * tmp671);
    const float tmp922 = (tmp921 * tmp917);
    const float tmp923 = (tmp921 * tmp918);
    const float tmp924 = (tmp921 * tmp919);
    const float tmp925 = (tmp921 * tmp920);
    const float tmp927 = (tmp893 + tmp922);
    const float tmp930 = (tmp896 + tmp923);
    const float tmp933 = (tmp899 + tmp924);
    const float tmp936 = (tmp902 + tmp925);
    const int32_t tmp8094 = (tmp7284 + tmp8198);
    S21_ch tmp6610 = tmp6414.children(tmp8094);
    device float* tmp6611 = tmp6610.get0(runtime_, mem_alloc_).val;
    const auto tmp939 = *tmp6611;
    device float* tmp6623 = tmp6610.get1(runtime_, mem_alloc_).val;
    const auto tmp941 = *tmp6623;
    const float tmp942 = (tmp641 * tmp644);
    const float tmp943 = (tmp942 * tmp939);
    const float tmp944 = (tmp942 * tmp941);
    const float tmp946 = (tmp912 + tmp943);
    const float tmp949 = (tmp915 + tmp944);
    const float tmp951 = (tmp939 * tmp868);
    const float tmp952 = (tmp939 * tmp727);
    const float tmp953 = (tmp941 * tmp868);
    const float tmp954 = (tmp941 * tmp727);
    const float tmp955 = (tmp942 * tmp671);
    const float tmp956 = (tmp955 * tmp951);
    const float tmp957 = (tmp955 * tmp952);
    const float tmp958 = (tmp955 * tmp953);
    const float tmp959 = (tmp955 * tmp954);
    const float tmp961 = (tmp927 + tmp956);
    const float tmp964 = (tmp930 + tmp957);
    const float tmp967 = (tmp933 + tmp958);
    const float tmp970 = (tmp936 + tmp959);
    S4 tmp6629 = tmp6392.get1(runtime_, mem_alloc_);
    S4_ch tmp6632 = tmp6629.children(tmp7236);
    device float* tmp6633 = tmp6632.get0(runtime_, mem_alloc_).val;
    *tmp6633 = tmp946;
    device float* tmp6643 = tmp6632.get1(runtime_, mem_alloc_).val;
    *tmp6643 = tmp949;
    S7 tmp6649 = tmp6392.get2(runtime_, mem_alloc_);
    S7_ch tmp6652 = tmp6649.children(tmp7236);
    device float* tmp6653 = tmp6652.get0(runtime_, mem_alloc_).val;
    *tmp6653 = tmp961;
    device float* tmp6663 = tmp6652.get1(runtime_, mem_alloc_).val;
    *tmp6663 = tmp964;
    device float* tmp6673 = tmp6652.get2(runtime_, mem_alloc_).val;
    *tmp6673 = tmp967;
    device float* tmp6683 = tmp6652.get3(runtime_, mem_alloc_).val;
    *tmp6683 = tmp970;
    const float tmp991 = (tmp946 * tmp990);
    const float tmp992 = (tmp949 * tmp990);
    const float tmp994 = (tmp609 + tmp991);
    *tmp6397 = tmp994;
    const float tmp997 = (tmp616 + tmp992);
    *tmp6407 = tmp997;
  } else {
  }
}

}  // namespace
kernel void mtl_k0003_substep_c4_0_0(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 16384;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0003_substep_c4_0_0_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0003_substep_c4_0_1(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 16384;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0003_substep_c4_0_1_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0003_substep_c4_0_2(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 16384;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0003_substep_c4_0_2_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0003_substep_c4_0_3(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 16384;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0003_substep_c4_0_3_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

