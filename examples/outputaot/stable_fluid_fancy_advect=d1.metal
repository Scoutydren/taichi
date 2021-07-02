#include <metal_stdlib>
#include <metal_compute>
using namespace metal;
namespace {
using byte = char;

template <typename T, typename G> T union_cast(G g) { static_assert(sizeof(T) == sizeof(G), "Size mismatch"); return *reinterpret_cast<thread const T *>(&g); } inline int ifloordiv(int lhs, int rhs) { const int intm = (lhs / rhs); return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs)) ? (intm - 1) : intm); } int32_t pow_i32(int32_t x, int32_t n) { int32_t tmp = x; int32_t ans = 1; while (n) { if (n & 1) ans *= tmp; tmp *= tmp; n >>= 1; } return ans; } float fatomic_fetch_add(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val + operand); ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_min(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val < operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_max(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val > operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } struct RandState { uint32_t seed; }; uint32_t metal_rand_u32(device RandState * state) { device uint *sp = (device uint *)&(state->seed); bool done = false; uint32_t nxt = 0; while (!done) { uint32_t o = *sp; nxt = o * 1103515245 + 12345; done = atomic_compare_exchange_weak_explicit( (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed, metal::memory_order_relaxed); } return nxt * 1000000007; } int32_t metal_rand_i32(device RandState * state) { return metal_rand_u32(state); } float metal_rand_f32(device RandState *state) { return metal_rand_u32(state) * (1.0f / 4294967296.0f); }

constant constexpr int kTaichiMaxNumIndices = 8; constant constexpr int kTaichiNumChunks = 1024; constant constexpr int kAlignment = 8; using PtrOffset = int32_t; struct MemoryAllocator { atomic_int next; constant constexpr static int kInitOffset = 8; static inline bool is_valid(PtrOffset v) { return v >= kInitOffset; } }; struct ListManagerData { int32_t element_stride = 0; int32_t log2_num_elems_per_chunk = 0; atomic_int next; atomic_int chunks[kTaichiNumChunks]; struct ReservedElemPtrOffset { public: ReservedElemPtrOffset() = default; explicit ReservedElemPtrOffset(PtrOffset v) : val_(v) { } inline bool is_valid() const { return is_valid(val_); } inline static bool is_valid(PtrOffset v) { return MemoryAllocator::is_valid(v); } inline PtrOffset value() const { return val_; } private: PtrOffset val_{0}; }; }; struct NodeManagerData { using ElemIndex = ListManagerData::ReservedElemPtrOffset; ListManagerData data_list; ListManagerData free_list; ListManagerData recycled_list; atomic_int free_list_used; int recycled_list_size_backup; }; struct SNodeMeta { enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3, Pointer = 4, BitStruct = 5, }; int32_t element_stride = 0; int32_t num_slots = 0; int32_t mem_offset_in_parent = 0; int32_t type = 0; }; struct SNodeExtractors { struct Extractor { int32_t start = 0; int32_t num_bits = 0; int32_t acc_offset = 0; int32_t num_elements = 0; }; Extractor extractors[kTaichiMaxNumIndices]; }; struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; }; struct ListgenElement { ElementCoords coords; int32_t mem_offset = 0; struct BelongedNodeManager { int32_t id = -1; NodeManagerData::ElemIndex elem_idx; }; BelongedNodeManager belonged_nodemgr; inline bool in_root_buffer() const { return belonged_nodemgr.id < 0; } };

struct Runtime {
  SNodeMeta snode_metas[24];
  SNodeExtractors snode_extractors[24];
  ListManagerData snode_lists[24];
  NodeManagerData snode_allocators[24];
  NodeManagerData::ElemIndex ambient_indices[24];
  uint32_t rand_seeds[65536];
};

[[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator *ma, int32_t size) { size = ((size + kAlignment - 1) / kAlignment) * kAlignment; return atomic_fetch_add_explicit(&ma->next, size, metal::memory_order_relaxed); } [[maybe_unused]] device char *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) { return reinterpret_cast<device char *>(ma + 1) + offs; } struct ListManager { using ReservedElemPtrOffset = ListManagerData::ReservedElemPtrOffset; device ListManagerData *lm_data; device MemoryAllocator *mem_alloc; inline int num_active() { return atomic_load_explicit(&(lm_data->next), metal::memory_order_relaxed); } inline void resize(int sz) { atomic_store_explicit(&(lm_data->next), sz, metal::memory_order_relaxed); } inline void clear() { resize(0); } ReservedElemPtrOffset reserve_new_elem() { const int elem_idx = atomic_fetch_add_explicit( &lm_data->next, 1, metal::memory_order_relaxed); const int chunk_idx = get_chunk_index(elem_idx); const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx); const auto offset = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return ReservedElemPtrOffset{offset}; } device char *append() { auto reserved = reserve_new_elem(); return get_ptr(reserved); } template <typename T> void append(thread const T &elem) { device char *ptr = append(); thread char *elem_ptr = (thread char *)(&elem); for (int i = 0; i < lm_data->element_stride; ++i) { *ptr = *elem_ptr; ++ptr; ++elem_ptr; } } device char *get_ptr(ReservedElemPtrOffset offs) { return mtl_memalloc_to_ptr(mem_alloc, offs.value()); } device char *get_ptr(int i) { const int chunk_idx = get_chunk_index(i); const PtrOffset chunk_ptr_offs = atomic_load_explicit( lm_data->chunks + chunk_idx, metal::memory_order_relaxed); return get_elem_from_chunk(i, chunk_ptr_offs); } template <typename T> T get(int i) { return *reinterpret_cast<device T *>(get_ptr(i)); } private: inline int get_chunk_index(int elem_idx) const { return elem_idx >> lm_data->log2_num_elems_per_chunk; } PtrOffset ensure_chunk(int chunk_idx) { PtrOffset offs = 0; const int chunk_bytes = (lm_data->element_stride << lm_data->log2_num_elems_per_chunk); while (true) { int stored = 0; const bool is_me = atomic_compare_exchange_weak_explicit( lm_data->chunks + chunk_idx, &stored, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes); atomic_store_explicit(lm_data->chunks + chunk_idx, offs, metal::memory_order_relaxed); break; } else if (stored > 1) { offs = stored; break; } } return offs; } PtrOffset get_elem_ptr_offs_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1); return chunk_ptr_offs + ((elem_idx & mask) * lm_data->element_stride); } device char *get_elem_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const auto offs = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return mtl_memalloc_to_ptr(mem_alloc, offs); } }; struct NodeManager { using ElemIndex = NodeManagerData::ElemIndex; device NodeManagerData *nm_data; device MemoryAllocator *mem_alloc; ElemIndex allocate() { ListManager free_list; free_list.lm_data = &(nm_data->free_list); free_list.mem_alloc = mem_alloc; ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; const int cur_used = atomic_fetch_add_explicit( &(nm_data->free_list_used), 1, metal::memory_order_relaxed); if (cur_used < free_list.num_active()) { return free_list.get<ElemIndex>(cur_used); } return data_list.reserve_new_elem(); } device byte *get(ElemIndex i) { ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; return data_list.get_ptr(i); } void recycle(ElemIndex i) { ListManager recycled_list; recycled_list.lm_data = &(nm_data->recycled_list); recycled_list.mem_alloc = mem_alloc; recycled_list.append(i); } }; class SNodeRep_dense { public: void init(device byte * addr) { addr_ = addr; } inline device byte *addr() { return addr_; } inline bool is_active(int) { return true; } inline void activate(int) { } inline void deactivate(int) { } private: device byte *addr_ = nullptr; }; using SNodeRep_root = SNodeRep_dense; class SNodeRep_bitmasked { public: constant static constexpr int kBitsPerMask = (sizeof(uint32_t) * 8); void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { device auto *ptr = to_bitmask_ptr(i); uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ((bits >> (i % kBitsPerMask)) & 1); } void activate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = (1 << (i % kBitsPerMask)); atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed); } void deactivate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = ~(1 << (i % kBitsPerMask)); atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed); } private: inline device atomic_uint *to_bitmask_ptr(int i) { return reinterpret_cast<device atomic_uint *>(addr_ + meta_offset_) + (i / kBitsPerMask); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_dynamic { public: void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { const auto n = atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); return i < n; } void activate(int i) { device auto *ptr = to_meta_ptr(); atomic_fetch_max_explicit(ptr, (i + 1), metal::memory_order_relaxed); return; } void deactivate() { device auto *ptr = to_meta_ptr(); atomic_store_explicit(ptr, 0, metal::memory_order_relaxed); } int append(int32_t data) { device auto *ptr = to_meta_ptr(); int me = atomic_fetch_add_explicit(ptr, 1, metal::memory_order_relaxed); *(reinterpret_cast<device int32_t *>(addr_) + me) = data; return me; } int length() { return atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); } private: inline device atomic_int *to_meta_ptr() { return reinterpret_cast<device atomic_int *>(addr_ + meta_offset_); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_pointer { public: using ElemIndex = NodeManagerData::ElemIndex; void init(device byte * addr, NodeManager nm, ElemIndex ambient_idx) { addr_ = addr; nm_ = nm; ambient_idx_ = ambient_idx; } device byte *child_or_ambient_addr(int i) { auto nm_idx = to_nodemgr_idx(addr_, i); nm_idx = nm_idx.is_valid() ? nm_idx : ambient_idx_; return nm_.get(nm_idx); } inline bool is_active(int i) { return is_active(addr_, i); } void activate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); auto nm_idx_val = atomic_load_explicit(nm_idx_ptr, metal::memory_order_relaxed); while (!ElemIndex::is_valid(nm_idx_val)) { nm_idx_val = 0; const bool is_me = atomic_compare_exchange_weak_explicit( nm_idx_ptr, &nm_idx_val, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { nm_idx_val = nm_.allocate().value(); atomic_store_explicit(nm_idx_ptr, nm_idx_val, metal::memory_order_relaxed); break; } else if (ElemIndex::is_valid(nm_idx_val)) { break; } } } void deactivate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); const auto old_nm_idx_val = atomic_exchange_explicit( nm_idx_ptr, 0, metal::memory_order_relaxed); const auto old_nm_idx = ElemIndex(old_nm_idx_val); if (!old_nm_idx.is_valid()) { return; } nm_.recycle(old_nm_idx); } static inline device atomic_int *to_nodemgr_idx_ptr(device byte * addr, int ch_i) { return reinterpret_cast<device atomic_int *>(addr + ch_i * sizeof(ElemIndex)); } static inline ElemIndex to_nodemgr_idx(device byte * addr, int ch_i) { device auto *ptr = to_nodemgr_idx_ptr(addr, ch_i); const auto v = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ElemIndex(v); } static bool is_active(device byte * addr, int ch_i) { return to_nodemgr_idx(addr, ch_i).is_valid(); } private: device byte *addr_; NodeManager nm_; ElemIndex ambient_idx_; }; [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return true; } else if (meta.type == SNodeMeta::Dynamic) { SNodeRep_dynamic rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Bitmasked) { SNodeRep_bitmasked rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Pointer) { return SNodeRep_pointer::is_active(addr, i); } return false; } [[maybe_unused]] void refine_coordinates( thread const ElementCoords &parent, device const SNodeExtractors &child_extrators, int l, thread ElementCoords *child) { for (int i = 0; i < kTaichiMaxNumIndices; ++i) { device const auto &ex = child_extrators.extractors[i]; const int mask = ((1 << ex.num_bits) - 1); const int addition = ((l >> ex.acc_offset) & mask); child->at[i] = ((parent.at[i] << ex.num_bits) | addition); } } [[maybe_unused]] device byte *mtl_lgen_snode_addr( thread const ListgenElement &lgen, device byte *root_addr, device Runtime *rtm, device MemoryAllocator *mem_alloc) { if (lgen.in_root_buffer()) { return root_addr + lgen.mem_offset; } NodeManager nm; nm.nm_data = (rtm->snode_allocators + lgen.belonged_nodemgr.id); nm.mem_alloc = mem_alloc; device byte *addr = nm.get(lgen.belonged_nodemgr.elem_idx); return addr + lgen.mem_offset; } [[maybe_unused]] void run_gc_compact_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, const int tid, const int grid_size) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_load_explicit(&(nm.nm_data->free_list_used), metal::memory_order_relaxed); int num_to_copy = 0; if (free_used * 2 > free_size) { num_to_copy = free_size - free_used; } else { num_to_copy = free_used; } const int offs = free_size - num_to_copy; using ElemIndex = NodeManager::ElemIndex; for (int ii = tid; ii < num_to_copy; ii += grid_size) { device auto *dest = reinterpret_cast<device ElemIndex *>(free_list.get_ptr(ii)); *dest = free_list.get<ElemIndex>(ii + offs); } } [[maybe_unused]] void run_gc_reset_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_exchange_explicit( &(nm.nm_data->free_list_used), 0, metal::memory_order_relaxed); int free_remaining = free_size - free_used; free_remaining = free_remaining > 0 ? free_remaining : 0; free_list.resize(free_remaining); nm.nm_data->recycled_list_size_backup = atomic_exchange_explicit( &(nm.nm_data->recycled_list.next), 0, metal::memory_order_relaxed); } struct GCMoveRecycledToFreeThreadParams { int thread_position_in_threadgroup; int threadgroup_position_in_grid; int threadgroups_per_grid; int threads_per_threadgroup; }; [[maybe_unused]] void run_gc_move_recycled_to_free( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, thread const GCMoveRecycledToFreeThreadParams &thparams) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; ListManager recycled_list; recycled_list.lm_data = &(nm.nm_data->recycled_list); recycled_list.mem_alloc = nm.mem_alloc; ListManager data_list; data_list.lm_data = &(nm.nm_data->data_list); data_list.mem_alloc = nm.mem_alloc; const int kInt32Stride = sizeof(int32_t); const int recycled_size = nm.nm_data->recycled_list_size_backup; using ElemIndex = NodeManager::ElemIndex; for (int ii = thparams.threadgroup_position_in_grid; ii < recycled_size; ii += thparams.threadgroups_per_grid) { const auto elem_idx = recycled_list.get<ElemIndex>(ii); device char *ptr = nm.get(elem_idx); device const char *ptr_end = ptr + data_list.lm_data->element_stride; const int ptr_mod = ((int64_t)(ptr) % kInt32Stride); if (ptr_mod) { device char *new_ptr = ptr + kInt32Stride - ptr_mod; if (thparams.thread_position_in_threadgroup == 0) { for (device char *p = ptr; p < new_ptr; ++p) { *p = 0; } } ptr = new_ptr; } ptr += (thparams.thread_position_in_threadgroup * kInt32Stride); while ((ptr + kInt32Stride) <= ptr_end) { *reinterpret_cast<device int32_t *>(ptr) = 0; ptr += (kInt32Stride * thparams.threads_per_threadgroup); } while (ptr < ptr_end) { *ptr = 0; ++ptr; } if (thparams.thread_position_in_threadgroup == 0) { free_list.append(elem_idx); } } }

struct SNodeBitPointer { device uint32_t *base; uint32_t offset; SNodeBitPointer(device byte * b, uint32_t o) : base((device uint32_t *)b), offset(o) { } }; template <typename C> C mtl_float_to_custom_int(float f) { const int32_t delta_bits = (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f); const float delta = union_cast<float>(delta_bits); return static_cast<C>(f + delta); } void mtl_set_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); bool ok = false; while (!ok) { P old_val = *(bp.base); P new_val = (old_val & (~mask)) | (value << bp.offset); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } } void mtl_set_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); atomic_store_explicit(atm_ptr, value, metal::memory_order_relaxed); } uint32_t mtl_atomic_add_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); P old_val = 0; bool ok = false; while (!ok) { old_val = *(bp.base); P new_val = old_val + (value << bp.offset); new_val = (old_val & (~mask)) | (new_val & mask); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } uint32_t mtl_atomic_add_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); return atomic_fetch_add_explicit(atm_ptr, value, metal::memory_order_relaxed); } namespace detail { template <bool Signed> struct SHRSelector { using type = int32_t; }; template <> struct SHRSelector<false> { using type = uint32_t; }; } template <typename C> C mtl_get_partial_bits(SNodeBitPointer bp, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const P phy_val = *(bp.base); using CSel = typename detail::SHRSelector<is_signed<C>::value>::type; const auto step1 = static_cast<CSel>(phy_val << (N - (bp.offset + bits))); return static_cast<C>(step1 >> (N - bits)); } template <typename C> C mtl_get_full_bits(SNodeBitPointer bp) { return static_cast<C>(*(bp.base)); }




struct S24 {
  // place
  constant static constexpr int stride = sizeof(float);

  S24(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S23_ch {
 public:
  S23_ch(device byte *a) : addr_(a) {}
  S24 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S24::stride;
 private:
  device byte *addr_;
};

struct S23 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S23_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S23(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S23_ch children(int i) {
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

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S22::stride;
 private:
  device byte *addr_;
};

struct S21 {
  // dense
  constant static constexpr int n = 524288;
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
  constant static constexpr int n = 524288;
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
  constant static constexpr int stride = sizeof(float);

  S18(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
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
  constant static constexpr int n = 524288;
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

class S14_ch {
 public:
  S14_ch(device byte *a) : addr_(a) {}
  S15 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S16 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S15::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S15::stride + S16::stride;
 private:
  device byte *addr_;
};

struct S14 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S14_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S14(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S14_ch children(int i) {
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


struct S13 {
  // place
  constant static constexpr int stride = sizeof(float);

  S13(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S12 {
  // place
  constant static constexpr int stride = sizeof(float);

  S12(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S11_ch {
 public:
  S11_ch(device byte *a) : addr_(a) {}
  S12 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S13 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S12::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S12::stride + S13::stride;
 private:
  device byte *addr_;
};

struct S11 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S11_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S11(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S11_ch children(int i) {
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


struct S7 {
  // place
  constant static constexpr int stride = sizeof(float);

  S7(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S6_ch {
 public:
  S6_ch(device byte *a) : addr_(a) {}
  S7 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S8 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S7::stride), rtm, ma};
  }

  S9 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S7::stride + S8::stride), rtm, ma};
  }

  S10 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S7::stride + S8::stride + S9::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S7::stride + S8::stride + S9::stride + S10::stride;
 private:
  device byte *addr_;
};

struct S6 {
  // dense
  constant static constexpr int n = 8388608;
  constant static constexpr int elem_stride = S6_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S6(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S6_ch children(int i) {
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


struct S5 {
  // place
  constant static constexpr int stride = sizeof(float);

  S5(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S4 {
  // place
  constant static constexpr int stride = sizeof(float);

  S4(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
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

  S4 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S2::stride + S3::stride), rtm, ma};
  }

  S5 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S2::stride + S3::stride + S4::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S2::stride + S3::stride + S4::stride + S5::stride;
 private:
  device byte *addr_;
};

struct S1 {
  // dense
  constant static constexpr int n = 8388608;
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

  S6 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride), rtm, ma};
  }

  S11 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride), rtm, ma};
  }

  S14 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride + S11::stride), rtm, ma};
  }

  S17 get4(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride + S11::stride + S14::stride), rtm, ma};
  }

  S19 get5(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride + S11::stride + S14::stride + S17::stride), rtm, ma};
  }

  S21 get6(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride + S11::stride + S14::stride + S17::stride + S19::stride), rtm, ma};
  }

  S23 get7(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S6::stride + S11::stride + S14::stride + S17::stride + S19::stride + S21::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S1::stride + S6::stride + S11::stride + S14::stride + S17::stride + S19::stride + S21::stride + S23::stride;
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

class mtl_k0007_advect_c4_3_args {
 public:
  explicit mtl_k0007_advect_c4_3_args(device byte* addr) : addr_(addr) {}
  device float* arg0() {
    // scalar, size=4 B
    return (device float*)(addr_ + 0);
  }
  device int32_t* arg1() {
    // scalar, size=4 B
    return (device int32_t*)(addr_ + 4);
  }
  device int32_t* arg2() {
    // scalar, size=4 B
    return (device int32_t*)(addr_ + 8);
  }
  
  int32_t extra_arg(int i, int j) {
    device int32_t* base = (device int32_t*)(addr_ + 12);
    return *(base + (i * 8) + j);
  }
 private:
  device byte* addr_;
};
void mtl_k0007_advect_c4_3_0_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* ctx_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  mtl_k0007_advect_c4_3_args kernel_ctx_(ctx_addr);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  constexpr int32_t tmp9410 = 10;
  constexpr int32_t tmp9178 = 4095;
  constexpr int32_t tmp9174 = 2047;
  constexpr float tmp97 = 0.0069849244;
  constexpr float tmp94 = 0.01;
  constexpr int32_t tmp9018 = 1023;
  constexpr int32_t tmp9014 = 511;
  constexpr int32_t tmp57 = 1;
  constexpr int32_t tmp50 = 596;
  constexpr int32_t tmp48 = 0;
  constexpr int32_t tmp46 = 416;
  constexpr float tmp36 = 597.0;
  constexpr float tmp34 = 417.0;
  constexpr float tmp28 = 1.0;
  constexpr float tmp23 = 0.5;
  const int tmp3 = linear_loop_idx_;
  constexpr int32_t tmp8948 = 12;
  const int32_t tmp8949 = (tmp3 >> tmp8948);
  const int32_t tmp8951 = (tmp8949 & tmp9174);
  const int32_t tmp8955 = (tmp3 & tmp9178);
  constexpr int32_t tmp13 = 1668;
  const int32_t tmp14 = -(tmp8951 < tmp13);
  constexpr int32_t tmp16 = 2388;
  const int32_t tmp17 = -(tmp8955 < tmp16);
  const int32_t tmp18 = (tmp14 & tmp17);
  if (tmp18) {
    const int32_t tmp20 = *kernel_ctx_.arg1();
    const int32_t tmp21 = *kernel_ctx_.arg2();
    const float tmp22 = static_cast<float>(tmp8951);
    const float tmp24 = (tmp22 + tmp23);
    const float tmp25 = static_cast<float>(tmp8955);
    const float tmp26 = (tmp25 + tmp23);
    const float tmp27 = static_cast<float>(tmp20);
    const float tmp29 = (tmp28 / tmp27);
    const float tmp30 = static_cast<float>(tmp21);
    const float tmp31 = (tmp28 / tmp30);
    const float tmp32 = (tmp24 * tmp29);
    const float tmp33 = (tmp26 * tmp31);
    const float tmp35 = (tmp32 * tmp34);
    const float tmp37 = (tmp33 * tmp36);
    const float tmp38 = (tmp35 - tmp23);
    const float tmp39 = (tmp37 - tmp23);
    const int32_t tmp40 = static_cast<int32_t>(tmp38);
    const int32_t tmp41 = static_cast<int32_t>(tmp39);
    const float tmp42 = static_cast<float>(tmp40);
    const float tmp43 = (tmp38 - tmp42);
    const float tmp44 = static_cast<float>(tmp41);
    const float tmp45 = (tmp39 - tmp44);
    const int32_t tmp47 =  min(tmp46, tmp40);
    const int32_t tmp49 =  max(tmp48, tmp47);
    const int32_t tmp51 =  min(tmp50, tmp41);
    const int32_t tmp52 =  max(tmp48, tmp51);
    S0 tmp8614(root_addr);
    S0_ch tmp8616 = tmp8614.children(tmp48);
    S14 tmp8617 = tmp8616.get3(runtime_, mem_alloc_);
    const int32_t tmp8959 = (tmp49 & tmp9014);
    const int32_t tmp8963 = (tmp52 & tmp9018);
    const int32_t tmp9411 = (tmp8959 << tmp9410);
    const int32_t tmp9187 = (tmp8963 + tmp9411);
    S14_ch tmp8621 = tmp8617.children(tmp9187);
    device float* tmp8622 = tmp8621.get0(runtime_, mem_alloc_).val;
    const auto tmp54 = *tmp8622;
    device float* tmp8634 = tmp8621.get1(runtime_, mem_alloc_).val;
    const auto tmp56 = *tmp8634;
    const int32_t tmp58 = (tmp40 + tmp57);
    const int32_t tmp59 =  min(tmp46, tmp58);
    const int32_t tmp60 =  max(tmp48, tmp59);
    const int32_t tmp8975 = (tmp60 & tmp9014);
    const int32_t tmp9413 = (tmp8975 << tmp9410);
    const int32_t tmp9203 = (tmp8963 + tmp9413);
    S14_ch tmp8645 = tmp8617.children(tmp9203);
    device float* tmp8646 = tmp8645.get0(runtime_, mem_alloc_).val;
    const auto tmp62 = *tmp8646;
    device float* tmp8658 = tmp8645.get1(runtime_, mem_alloc_).val;
    const auto tmp64 = *tmp8658;
    const int32_t tmp65 = (tmp41 + tmp57);
    const int32_t tmp66 =  min(tmp50, tmp65);
    const int32_t tmp67 =  max(tmp48, tmp66);
    const int32_t tmp8995 = (tmp67 & tmp9018);
    const int32_t tmp9219 = (tmp8995 + tmp9411);
    S14_ch tmp8669 = tmp8617.children(tmp9219);
    device float* tmp8670 = tmp8669.get0(runtime_, mem_alloc_).val;
    const auto tmp69 = *tmp8670;
    device float* tmp8682 = tmp8669.get1(runtime_, mem_alloc_).val;
    const auto tmp71 = *tmp8682;
    const int32_t tmp9235 = (tmp8995 + tmp9413);
    S14_ch tmp8693 = tmp8617.children(tmp9235);
    device float* tmp8694 = tmp8693.get0(runtime_, mem_alloc_).val;
    const auto tmp73 = *tmp8694;
    device float* tmp8706 = tmp8693.get1(runtime_, mem_alloc_).val;
    const auto tmp75 = *tmp8706;
    const float tmp76 = (tmp62 - tmp54);
    const float tmp77 = (tmp43 * tmp76);
    const float tmp78 = (tmp54 + tmp77);
    const float tmp79 = (tmp64 - tmp56);
    const float tmp80 = (tmp43 * tmp79);
    const float tmp81 = (tmp56 + tmp80);
    const float tmp82 = (tmp73 - tmp69);
    const float tmp83 = (tmp43 * tmp82);
    const float tmp84 = (tmp69 + tmp83);
    const float tmp85 = (tmp75 - tmp71);
    const float tmp86 = (tmp43 * tmp85);
    const float tmp87 = (tmp71 + tmp86);
    const float tmp88 = (tmp84 - tmp78);
    const float tmp89 = (tmp45 * tmp88);
    const float tmp90 = (tmp78 + tmp89);
    const float tmp91 = (tmp87 - tmp81);
    const float tmp92 = (tmp45 * tmp91);
    const float tmp93 = (tmp81 + tmp92);
    const float tmp95 = (tmp90 * tmp94);
    const float tmp96 = (tmp32 - tmp95);
    const float tmp98 = (tmp93 * tmp97);
    const float tmp99 = (tmp33 - tmp98);
    const float tmp100 = (tmp96 * tmp27);
    const float tmp101 = (tmp99 * tmp30);
    const float tmp102 = (tmp100 - tmp23);
    const float tmp103 = (tmp101 - tmp23);
    const int32_t tmp104 = static_cast<int32_t>(tmp102);
    const int32_t tmp105 = static_cast<int32_t>(tmp103);
    const float tmp106 = static_cast<float>(tmp104);
    const float tmp107 = (tmp102 - tmp106);
    const float tmp108 = static_cast<float>(tmp105);
    const float tmp109 = (tmp103 - tmp108);
    const int32_t tmp110 = (tmp20 - tmp57);
    const int32_t tmp111 =  min(tmp110, tmp104);
    const int32_t tmp112 =  max(tmp48, tmp111);
    const int32_t tmp113 = (tmp21 - tmp57);
    const int32_t tmp114 =  min(tmp113, tmp105);
    const int32_t tmp115 =  max(tmp48, tmp114);
    S6 tmp8713 = tmp8616.get1(runtime_, mem_alloc_);
    const int32_t tmp9023 = (tmp112 & tmp9174);
    const int32_t tmp9027 = (tmp115 & tmp9178);
    const int32_t tmp9415 = (tmp9023 << tmp8948);
    const int32_t tmp9251 = (tmp9027 + tmp9415);
    S6_ch tmp8717 = tmp8713.children(tmp9251);
    device float* tmp8718 = tmp8717.get0(runtime_, mem_alloc_).val;
    const auto tmp117 = *tmp8718;
    device float* tmp8730 = tmp8717.get1(runtime_, mem_alloc_).val;
    const auto tmp119 = *tmp8730;
    device float* tmp8742 = tmp8717.get2(runtime_, mem_alloc_).val;
    const auto tmp121 = *tmp8742;
    device float* tmp8754 = tmp8717.get3(runtime_, mem_alloc_).val;
    const auto tmp123 = *tmp8754;
    const int32_t tmp124 = (tmp104 + tmp57);
    const int32_t tmp125 =  min(tmp110, tmp124);
    const int32_t tmp126 =  max(tmp48, tmp125);
    const int32_t tmp9055 = (tmp126 & tmp9174);
    const int32_t tmp9417 = (tmp9055 << tmp8948);
    const int32_t tmp9283 = (tmp9027 + tmp9417);
    S6_ch tmp8765 = tmp8713.children(tmp9283);
    device float* tmp8766 = tmp8765.get0(runtime_, mem_alloc_).val;
    const auto tmp128 = *tmp8766;
    device float* tmp8778 = tmp8765.get1(runtime_, mem_alloc_).val;
    const auto tmp130 = *tmp8778;
    device float* tmp8790 = tmp8765.get2(runtime_, mem_alloc_).val;
    const auto tmp132 = *tmp8790;
    device float* tmp8802 = tmp8765.get3(runtime_, mem_alloc_).val;
    const auto tmp134 = *tmp8802;
    const int32_t tmp135 = (tmp105 + tmp57);
    const int32_t tmp136 =  min(tmp113, tmp135);
    const int32_t tmp137 =  max(tmp48, tmp136);
    const int32_t tmp9091 = (tmp137 & tmp9178);
    const int32_t tmp9315 = (tmp9091 + tmp9415);
    S6_ch tmp8813 = tmp8713.children(tmp9315);
    device float* tmp8814 = tmp8813.get0(runtime_, mem_alloc_).val;
    const auto tmp139 = *tmp8814;
    device float* tmp8826 = tmp8813.get1(runtime_, mem_alloc_).val;
    const auto tmp141 = *tmp8826;
    device float* tmp8838 = tmp8813.get2(runtime_, mem_alloc_).val;
    const auto tmp143 = *tmp8838;
    device float* tmp8850 = tmp8813.get3(runtime_, mem_alloc_).val;
    const auto tmp145 = *tmp8850;
    const int32_t tmp9347 = (tmp9091 + tmp9417);
    S6_ch tmp8861 = tmp8713.children(tmp9347);
    device float* tmp8862 = tmp8861.get0(runtime_, mem_alloc_).val;
    const auto tmp147 = *tmp8862;
    device float* tmp8874 = tmp8861.get1(runtime_, mem_alloc_).val;
    const auto tmp149 = *tmp8874;
    device float* tmp8886 = tmp8861.get2(runtime_, mem_alloc_).val;
    const auto tmp151 = *tmp8886;
    device float* tmp8898 = tmp8861.get3(runtime_, mem_alloc_).val;
    const auto tmp153 = *tmp8898;
    const float tmp154 = (tmp128 - tmp117);
    const float tmp155 = (tmp107 * tmp154);
    const float tmp156 = (tmp117 + tmp155);
    const float tmp157 = (tmp130 - tmp119);
    const float tmp158 = (tmp107 * tmp157);
    const float tmp159 = (tmp119 + tmp158);
    const float tmp160 = (tmp132 - tmp121);
    const float tmp161 = (tmp107 * tmp160);
    const float tmp162 = (tmp121 + tmp161);
    const float tmp163 = (tmp134 - tmp123);
    const float tmp164 = (tmp107 * tmp163);
    const float tmp165 = (tmp123 + tmp164);
    const float tmp166 = (tmp147 - tmp139);
    const float tmp167 = (tmp107 * tmp166);
    const float tmp168 = (tmp139 + tmp167);
    const float tmp169 = (tmp149 - tmp141);
    const float tmp170 = (tmp107 * tmp169);
    const float tmp171 = (tmp141 + tmp170);
    const float tmp172 = (tmp151 - tmp143);
    const float tmp173 = (tmp107 * tmp172);
    const float tmp174 = (tmp143 + tmp173);
    const float tmp175 = (tmp153 - tmp145);
    const float tmp176 = (tmp107 * tmp175);
    const float tmp177 = (tmp145 + tmp176);
    const float tmp178 = (tmp168 - tmp156);
    const float tmp179 = (tmp109 * tmp178);
    const float tmp180 = (tmp156 + tmp179);
    const float tmp181 = (tmp171 - tmp159);
    const float tmp182 = (tmp109 * tmp181);
    const float tmp183 = (tmp159 + tmp182);
    const float tmp184 = (tmp174 - tmp162);
    const float tmp185 = (tmp109 * tmp184);
    const float tmp186 = (tmp162 + tmp185);
    const float tmp187 = (tmp177 - tmp165);
    const float tmp188 = (tmp109 * tmp187);
    const float tmp189 = (tmp165 + tmp188);
    const float tmp190 = *kernel_ctx_.arg0();
    const float tmp191 = (tmp190 * tmp94);
    const float tmp192 = (tmp191 + tmp28);
    const float tmp193 = (tmp180 / tmp192);
    const float tmp194 = (tmp183 / tmp192);
    const float tmp195 = (tmp186 / tmp192);
    const float tmp196 = (tmp189 / tmp192);
    S1 tmp8905 = tmp8616.get0(runtime_, mem_alloc_);
    const int32_t tmp9419 = (tmp8951 << tmp8948);
    const int32_t tmp9379 = (tmp8955 + tmp9419);
    S1_ch tmp8909 = tmp8905.children(tmp9379);
    device float* tmp8910 = tmp8909.get0(runtime_, mem_alloc_).val;
    *tmp8910 = tmp193;
    device float* tmp8922 = tmp8909.get1(runtime_, mem_alloc_).val;
    *tmp8922 = tmp194;
    device float* tmp8934 = tmp8909.get2(runtime_, mem_alloc_).val;
    *tmp8934 = tmp195;
    device float* tmp8946 = tmp8909.get3(runtime_, mem_alloc_).val;
    *tmp8946 = tmp196;
  } else {
  }
}

}  // namespace
kernel void mtl_k0007_advect_c4_3_0(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* ctx_addr [[buffer(2)]],
    device byte* runtime_addr [[buffer(3)]],
    device byte* print_assert_addr [[buffer(4)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 8388608;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0007_advect_c4_3_0_func(root_addr, global_tmps_addr, ctx_addr, runtime_addr, print_assert_addr, ii);
  }
}

