#include <metal_stdlib>
#include <metal_compute>
using namespace metal;
namespace {
using byte = char;

template <typename T, typename G> T union_cast(G g) { static_assert(sizeof(T) == sizeof(G), "Size mismatch"); return *reinterpret_cast<thread const T *>(&g); } inline int ifloordiv(int lhs, int rhs) { const int intm = (lhs / rhs); return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs)) ? (intm - 1) : intm); } int32_t pow_i32(int32_t x, int32_t n) { int32_t tmp = x; int32_t ans = 1; while (n) { if (n & 1) ans *= tmp; tmp *= tmp; n >>= 1; } return ans; } float fatomic_fetch_add(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val + operand); ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_min(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val < operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_max(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val > operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } struct RandState { uint32_t seed; }; uint32_t metal_rand_u32(device RandState * state) { device uint *sp = (device uint *)&(state->seed); bool done = false; uint32_t nxt = 0; while (!done) { uint32_t o = *sp; nxt = o * 1103515245 + 12345; done = atomic_compare_exchange_weak_explicit( (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed, metal::memory_order_relaxed); } return nxt * 1000000007; } int32_t metal_rand_i32(device RandState * state) { return metal_rand_u32(state); } float metal_rand_f32(device RandState *state) { return metal_rand_u32(state) * (1.0f / 4294967296.0f); }

constant constexpr int kTaichiMaxNumIndices = 8; constant constexpr int kTaichiNumChunks = 1024; constant constexpr int kAlignment = 8; using PtrOffset = int32_t; struct MemoryAllocator { atomic_int next; constant constexpr static int kInitOffset = 8; static inline bool is_valid(PtrOffset v) { return v >= kInitOffset; } }; struct ListManagerData { int32_t element_stride = 0; int32_t log2_num_elems_per_chunk = 0; atomic_int next; atomic_int chunks[kTaichiNumChunks]; struct ReservedElemPtrOffset { public: ReservedElemPtrOffset() = default; explicit ReservedElemPtrOffset(PtrOffset v) : val_(v) { } inline bool is_valid() const { return is_valid(val_); } inline static bool is_valid(PtrOffset v) { return MemoryAllocator::is_valid(v); } inline PtrOffset value() const { return val_; } private: PtrOffset val_{0}; }; }; struct NodeManagerData { using ElemIndex = ListManagerData::ReservedElemPtrOffset; ListManagerData data_list; ListManagerData free_list; ListManagerData recycled_list; atomic_int free_list_used; int recycled_list_size_backup; }; struct SNodeMeta { enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3, Pointer = 4, BitStruct = 5, }; int32_t element_stride = 0; int32_t num_slots = 0; int32_t mem_offset_in_parent = 0; int32_t type = 0; }; struct SNodeExtractors { struct Extractor { int32_t start = 0; int32_t num_bits = 0; int32_t acc_offset = 0; int32_t num_elements_from_root = 0; }; Extractor extractors[kTaichiMaxNumIndices]; }; struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; }; struct ListgenElement { ElementCoords coords; int32_t mem_offset = 0; struct BelongedNodeManager { int32_t id = -1; NodeManagerData::ElemIndex elem_idx; }; BelongedNodeManager belonged_nodemgr; inline bool in_root_buffer() const { return belonged_nodemgr.id < 0; } };

struct Runtime {
  SNodeMeta snode_metas[18];
  SNodeExtractors snode_extractors[18];
  ListManagerData snode_lists[18];
  NodeManagerData snode_allocators[18];
  NodeManagerData::ElemIndex ambient_indices[18];
  uint32_t rand_seeds[65536];
};

[[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator *ma, int32_t size) { size = ((size + kAlignment - 1) / kAlignment) * kAlignment; return atomic_fetch_add_explicit(&ma->next, size, metal::memory_order_relaxed); } [[maybe_unused]] device char *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) { return reinterpret_cast<device char *>(ma + 1) + offs; } struct ListManager { using ReservedElemPtrOffset = ListManagerData::ReservedElemPtrOffset; device ListManagerData *lm_data; device MemoryAllocator *mem_alloc; inline int num_active() { return atomic_load_explicit(&(lm_data->next), metal::memory_order_relaxed); } inline void resize(int sz) { atomic_store_explicit(&(lm_data->next), sz, metal::memory_order_relaxed); } inline void clear() { resize(0); } ReservedElemPtrOffset reserve_new_elem() { const int elem_idx = atomic_fetch_add_explicit( &lm_data->next, 1, metal::memory_order_relaxed); const int chunk_idx = get_chunk_index(elem_idx); const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx); const auto offset = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return ReservedElemPtrOffset{offset}; } device char *append() { auto reserved = reserve_new_elem(); return get_ptr(reserved); } template <typename T> void append(thread const T &elem) { device char *ptr = append(); thread char *elem_ptr = (thread char *)(&elem); for (int i = 0; i < lm_data->element_stride; ++i) { *ptr = *elem_ptr; ++ptr; ++elem_ptr; } } device char *get_ptr(ReservedElemPtrOffset offs) { return mtl_memalloc_to_ptr(mem_alloc, offs.value()); } device char *get_ptr(int i) { const int chunk_idx = get_chunk_index(i); const PtrOffset chunk_ptr_offs = atomic_load_explicit( lm_data->chunks + chunk_idx, metal::memory_order_relaxed); return get_elem_from_chunk(i, chunk_ptr_offs); } template <typename T> T get(int i) { return *reinterpret_cast<device T *>(get_ptr(i)); } private: inline int get_chunk_index(int elem_idx) const { return elem_idx >> lm_data->log2_num_elems_per_chunk; } PtrOffset ensure_chunk(int chunk_idx) { PtrOffset offs = 0; const int chunk_bytes = (lm_data->element_stride << lm_data->log2_num_elems_per_chunk); while (true) { int stored = 0; const bool is_me = atomic_compare_exchange_weak_explicit( lm_data->chunks + chunk_idx, &stored, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes); atomic_store_explicit(lm_data->chunks + chunk_idx, offs, metal::memory_order_relaxed); break; } else if (stored > 1) { offs = stored; break; } } return offs; } PtrOffset get_elem_ptr_offs_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1); return chunk_ptr_offs + ((elem_idx & mask) * lm_data->element_stride); } device char *get_elem_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const auto offs = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return mtl_memalloc_to_ptr(mem_alloc, offs); } }; struct NodeManager { using ElemIndex = NodeManagerData::ElemIndex; device NodeManagerData *nm_data; device MemoryAllocator *mem_alloc; ElemIndex allocate() { ListManager free_list; free_list.lm_data = &(nm_data->free_list); free_list.mem_alloc = mem_alloc; ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; const int cur_used = atomic_fetch_add_explicit( &(nm_data->free_list_used), 1, metal::memory_order_relaxed); if (cur_used < free_list.num_active()) { return free_list.get<ElemIndex>(cur_used); } return data_list.reserve_new_elem(); } device byte *get(ElemIndex i) { ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; return data_list.get_ptr(i); } void recycle(ElemIndex i) { ListManager recycled_list; recycled_list.lm_data = &(nm_data->recycled_list); recycled_list.mem_alloc = mem_alloc; recycled_list.append(i); } }; class SNodeRep_dense { public: void init(device byte * addr) { addr_ = addr; } inline device byte *addr() { return addr_; } inline bool is_active(int) { return true; } inline void activate(int) { } inline void deactivate(int) { } private: device byte *addr_ = nullptr; }; using SNodeRep_root = SNodeRep_dense; class SNodeRep_bitmasked { public: constant static constexpr int kBitsPerMask = (sizeof(uint32_t) * 8); void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { device auto *ptr = to_bitmask_ptr(i); uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ((bits >> (i % kBitsPerMask)) & 1); } void activate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = (1 << (i % kBitsPerMask)); atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed); } void deactivate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = ~(1 << (i % kBitsPerMask)); atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed); } private: inline device atomic_uint *to_bitmask_ptr(int i) { return reinterpret_cast<device atomic_uint *>(addr_ + meta_offset_) + (i / kBitsPerMask); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_dynamic { public: void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { const auto n = atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); return i < n; } void activate(int i) { device auto *ptr = to_meta_ptr(); atomic_fetch_max_explicit(ptr, (i + 1), metal::memory_order_relaxed); return; } void deactivate() { device auto *ptr = to_meta_ptr(); atomic_store_explicit(ptr, 0, metal::memory_order_relaxed); } int append(int32_t data) { device auto *ptr = to_meta_ptr(); int me = atomic_fetch_add_explicit(ptr, 1, metal::memory_order_relaxed); *(reinterpret_cast<device int32_t *>(addr_) + me) = data; return me; } int length() { return atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); } private: inline device atomic_int *to_meta_ptr() { return reinterpret_cast<device atomic_int *>(addr_ + meta_offset_); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_pointer { public: using ElemIndex = NodeManagerData::ElemIndex; void init(device byte * addr, NodeManager nm, ElemIndex ambient_idx) { addr_ = addr; nm_ = nm; ambient_idx_ = ambient_idx; } device byte *child_or_ambient_addr(int i) { auto nm_idx = to_nodemgr_idx(addr_, i); nm_idx = nm_idx.is_valid() ? nm_idx : ambient_idx_; return nm_.get(nm_idx); } inline bool is_active(int i) { return is_active(addr_, i); } void activate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); auto nm_idx_val = atomic_load_explicit(nm_idx_ptr, metal::memory_order_relaxed); while (!ElemIndex::is_valid(nm_idx_val)) { nm_idx_val = 0; const bool is_me = atomic_compare_exchange_weak_explicit( nm_idx_ptr, &nm_idx_val, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { nm_idx_val = nm_.allocate().value(); atomic_store_explicit(nm_idx_ptr, nm_idx_val, metal::memory_order_relaxed); break; } else if (ElemIndex::is_valid(nm_idx_val)) { break; } } } void deactivate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); const auto old_nm_idx_val = atomic_exchange_explicit( nm_idx_ptr, 0, metal::memory_order_relaxed); const auto old_nm_idx = ElemIndex(old_nm_idx_val); if (!old_nm_idx.is_valid()) { return; } nm_.recycle(old_nm_idx); } static inline device atomic_int *to_nodemgr_idx_ptr(device byte * addr, int ch_i) { return reinterpret_cast<device atomic_int *>(addr + ch_i * sizeof(ElemIndex)); } static inline ElemIndex to_nodemgr_idx(device byte * addr, int ch_i) { device auto *ptr = to_nodemgr_idx_ptr(addr, ch_i); const auto v = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ElemIndex(v); } static bool is_active(device byte * addr, int ch_i) { return to_nodemgr_idx(addr, ch_i).is_valid(); } private: device byte *addr_; NodeManager nm_; ElemIndex ambient_idx_; }; [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return true; } else if (meta.type == SNodeMeta::Dynamic) { SNodeRep_dynamic rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Bitmasked) { SNodeRep_bitmasked rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Pointer) { return SNodeRep_pointer::is_active(addr, i); } return false; } [[maybe_unused]] void refine_coordinates( thread const ElementCoords &parent, device const SNodeExtractors &child_extrators, int l, thread ElementCoords *child) { for (int i = 0; i < kTaichiMaxNumIndices; ++i) { device const auto &ex = child_extrators.extractors[i]; const int mask = ((1 << ex.num_bits) - 1); const int addition = ((l >> ex.acc_offset) & mask); child->at[i] = ((parent.at[i] << ex.num_bits) | addition); } } [[maybe_unused]] device byte *mtl_lgen_snode_addr( thread const ListgenElement &lgen, device byte *root_addr, device Runtime *rtm, device MemoryAllocator *mem_alloc) { if (lgen.in_root_buffer()) { return root_addr + lgen.mem_offset; } NodeManager nm; nm.nm_data = (rtm->snode_allocators + lgen.belonged_nodemgr.id); nm.mem_alloc = mem_alloc; device byte *addr = nm.get(lgen.belonged_nodemgr.elem_idx); return addr + lgen.mem_offset; } [[maybe_unused]] void run_gc_compact_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, const int tid, const int grid_size) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_load_explicit(&(nm.nm_data->free_list_used), metal::memory_order_relaxed); int num_to_copy = 0; if (free_used * 2 > free_size) { num_to_copy = free_size - free_used; } else { num_to_copy = free_used; } const int offs = free_size - num_to_copy; using ElemIndex = NodeManager::ElemIndex; for (int ii = tid; ii < num_to_copy; ii += grid_size) { device auto *dest = reinterpret_cast<device ElemIndex *>(free_list.get_ptr(ii)); *dest = free_list.get<ElemIndex>(ii + offs); } } [[maybe_unused]] void run_gc_reset_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_exchange_explicit( &(nm.nm_data->free_list_used), 0, metal::memory_order_relaxed); int free_remaining = free_size - free_used; free_remaining = free_remaining > 0 ? free_remaining : 0; free_list.resize(free_remaining); nm.nm_data->recycled_list_size_backup = atomic_exchange_explicit( &(nm.nm_data->recycled_list.next), 0, metal::memory_order_relaxed); } struct GCMoveRecycledToFreeThreadParams { int thread_position_in_threadgroup; int threadgroup_position_in_grid; int threadgroups_per_grid; int threads_per_threadgroup; }; [[maybe_unused]] void run_gc_move_recycled_to_free( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, thread const GCMoveRecycledToFreeThreadParams &thparams) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; ListManager recycled_list; recycled_list.lm_data = &(nm.nm_data->recycled_list); recycled_list.mem_alloc = nm.mem_alloc; ListManager data_list; data_list.lm_data = &(nm.nm_data->data_list); data_list.mem_alloc = nm.mem_alloc; const int kInt32Stride = sizeof(int32_t); const int recycled_size = nm.nm_data->recycled_list_size_backup; using ElemIndex = NodeManager::ElemIndex; for (int ii = thparams.threadgroup_position_in_grid; ii < recycled_size; ii += thparams.threadgroups_per_grid) { const auto elem_idx = recycled_list.get<ElemIndex>(ii); device char *ptr = nm.get(elem_idx); device const char *ptr_end = ptr + data_list.lm_data->element_stride; const int ptr_mod = ((int64_t)(ptr) % kInt32Stride); if (ptr_mod) { device char *new_ptr = ptr + kInt32Stride - ptr_mod; if (thparams.thread_position_in_threadgroup == 0) { for (device char *p = ptr; p < new_ptr; ++p) { *p = 0; } } ptr = new_ptr; } ptr += (thparams.thread_position_in_threadgroup * kInt32Stride); while ((ptr + kInt32Stride) <= ptr_end) { *reinterpret_cast<device int32_t *>(ptr) = 0; ptr += (kInt32Stride * thparams.threads_per_threadgroup); } while (ptr < ptr_end) { *ptr = 0; ++ptr; } if (thparams.thread_position_in_threadgroup == 0) { free_list.append(elem_idx); } } }

struct SNodeBitPointer { device uint32_t *base; uint32_t offset; SNodeBitPointer(device byte * b, uint32_t o) : base((device uint32_t *)b), offset(o) { } }; template <typename C> C mtl_float_to_custom_int(float f) { const int32_t delta_bits = (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f); const float delta = union_cast<float>(delta_bits); return static_cast<C>(f + delta); } void mtl_set_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); bool ok = false; while (!ok) { P old_val = *(bp.base); P new_val = (old_val & (~mask)) | (value << bp.offset); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } } void mtl_set_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); atomic_store_explicit(atm_ptr, value, metal::memory_order_relaxed); } uint32_t mtl_atomic_add_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); P old_val = 0; bool ok = false; while (!ok) { old_val = *(bp.base); P new_val = old_val + (value << bp.offset); new_val = (old_val & (~mask)) | (new_val & mask); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } uint32_t mtl_atomic_add_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); return atomic_fetch_add_explicit(atm_ptr, value, metal::memory_order_relaxed); } namespace detail { template <bool Signed> struct SHRSelector { using type = int32_t; }; template <> struct SHRSelector<false> { using type = uint32_t; }; } template <typename C> C mtl_get_partial_bits(SNodeBitPointer bp, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const P phy_val = *(bp.base); using CSel = typename detail::SHRSelector<is_signed<C>::value>::type; const auto step1 = static_cast<CSel>(phy_val << (N - (bp.offset + bits))); return static_cast<C>(step1 >> (N - bits)); } template <typename C> C mtl_get_full_bits(SNodeBitPointer bp) { return static_cast<C>(*(bp.base)); }




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
  constant static constexpr int n = 16384;
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

class S12_ch {
 public:
  S12_ch(device byte *a) : addr_(a) {}
  S13 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S13::stride;
 private:
  device byte *addr_;
};

struct S12 {
  // dense
  constant static constexpr int n = 8192;
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
  constant static constexpr int n = 8192;
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
  constant static constexpr int n = 8192;
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
  constant static constexpr int n = 8192;
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

  S14 get4(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride), rtm, ma};
  }

  S17 get5(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S4::stride + S7::stride + S12::stride + S14::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S1::stride + S4::stride + S7::stride + S12::stride + S14::stride + S17::stride;
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

void mtl_k0004_substep_c4_0_0_func(
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
  constexpr int32_t tmp5142 = 7;
  const int32_t tmp5143 = (tmp3 >> tmp5142);
  constexpr float tmp14 = 0.0;
  S0 tmp4146(root_addr);
  constexpr int32_t tmp5770 = 0;
  S0_ch tmp4148 = tmp4146.children(tmp5770);
  S14 tmp4149 = tmp4148.get4(runtime_, mem_alloc_);
  constexpr int32_t tmp6421 = 127;
  const int32_t tmp6404 = (tmp5143 & tmp6421);
  const int32_t tmp6406 = (tmp3 & tmp6421);
  const int32_t tmp6428 = (tmp6404 << tmp5142);
  const int32_t tmp5777 = (tmp6406 + tmp6428);
  S14_ch tmp4153 = tmp4149.children(tmp5777);
  device float* tmp4154 = tmp4153.get0(runtime_, mem_alloc_).val;
  *tmp4154 = tmp14;
  device float* tmp4166 = tmp4153.get1(runtime_, mem_alloc_).val;
  *tmp4166 = tmp14;
  S17 tmp4173 = tmp4148.get5(runtime_, mem_alloc_);
  S17_ch tmp4177 = tmp4173.children(tmp5777);
  device float* tmp4178 = tmp4177.get0(runtime_, mem_alloc_).val;
  *tmp4178 = tmp14;
}

void mtl_k0004_substep_c4_0_1_func(
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
  const int tmp22 = linear_loop_idx_;
  S0 tmp4181(root_addr);
  constexpr int32_t tmp5794 = 0;
  S0_ch tmp4183 = tmp4181.children(tmp5794);
  S1 tmp4184 = tmp4183.get0(runtime_, mem_alloc_);
  constexpr int32_t tmp6423 = 8191;
  const int32_t tmp6408 = (tmp22 & tmp6423);
  constexpr int32_t tmp5796 = 1;
  S1_ch tmp4187 = tmp4184.children(tmp6408);
  device float* tmp4188 = tmp4187.get0(runtime_, mem_alloc_).val;
  const auto tmp29 = *tmp4188;
  constexpr float tmp30 = 128.0;
  const float tmp31 = (tmp29 * tmp30);
  device float* tmp4198 = tmp4187.get1(runtime_, mem_alloc_).val;
  const auto tmp33 = *tmp4198;
  const float tmp34 = (tmp33 * tmp30);
  constexpr float tmp35 = 0.5;
  const float tmp36 = (tmp31 - tmp35);
  const int32_t tmp37 = static_cast<int32_t>(tmp36);
  const float tmp38 = (tmp34 - tmp35);
  const int32_t tmp39 = static_cast<int32_t>(tmp38);
  const float tmp40 = static_cast<float>(tmp37);
  const float tmp41 = (tmp31 - tmp40);
  const float tmp42 = static_cast<float>(tmp39);
  const float tmp43 = (tmp34 - tmp42);
  constexpr float tmp44 = 1.5;
  const float tmp45 = (tmp44 - tmp41);
  const float tmp46 = (tmp45 * tmp45);
  const float tmp47 = (tmp46 * tmp35);
  const float tmp48 = (tmp44 - tmp43);
  const float tmp49 = (tmp48 * tmp48);
  const float tmp50 = (tmp49 * tmp35);
  constexpr float tmp51 = 1.0;
  const float tmp52 = (tmp41 - tmp51);
  const float tmp53 = (tmp52 * tmp52);
  constexpr float tmp54 = 0.75;
  const float tmp55 = (tmp54 - tmp53);
  const float tmp56 = (tmp43 - tmp51);
  const float tmp57 = (tmp56 * tmp56);
  const float tmp58 = (tmp54 - tmp57);
  const float tmp59 = (tmp41 - tmp35);
  const float tmp60 = (tmp59 * tmp59);
  const float tmp61 = (tmp60 * tmp35);
  const float tmp62 = (tmp43 - tmp35);
  const float tmp63 = (tmp62 * tmp62);
  const float tmp64 = (tmp63 * tmp35);
  S12 tmp4204 = tmp4183.get3(runtime_, mem_alloc_);
  S12_ch tmp4207 = tmp4204.children(tmp6408);
  device float* tmp4208 = tmp4207.get0(runtime_, mem_alloc_).val;
  const auto tmp66 = *tmp4208;
  const float tmp67 = (tmp66 - tmp51);
  constexpr float tmp68 = -0.08;
  const float tmp69 = (tmp67 * tmp68);
  S7 tmp4214 = tmp4183.get2(runtime_, mem_alloc_);
  S7_ch tmp4217 = tmp4214.children(tmp6408);
  device float* tmp4218 = tmp4217.get0(runtime_, mem_alloc_).val;
  const auto tmp71 = *tmp4218;
  constexpr float tmp72 = 1.5258789e-05;
  const float tmp73 = (tmp71 * tmp72);
  const float tmp74 = (tmp69 + tmp73);
  device float* tmp4228 = tmp4217.get1(runtime_, mem_alloc_).val;
  const auto tmp76 = *tmp4228;
  const float tmp77 = (tmp76 * tmp72);
  device float* tmp4238 = tmp4217.get2(runtime_, mem_alloc_).val;
  const auto tmp79 = *tmp4238;
  const float tmp80 = (tmp79 * tmp72);
  device float* tmp4248 = tmp4217.get3(runtime_, mem_alloc_).val;
  const auto tmp82 = *tmp4248;
  const float tmp83 = (tmp82 * tmp72);
  const float tmp84 = (tmp69 + tmp83);
  constexpr float tmp85 = 0.0;
  const float tmp86 = (tmp85 - tmp41);
  constexpr float tmp87 = 0.0078125;
  const float tmp88 = (tmp86 * tmp87);
  const float tmp89 = (tmp85 - tmp43);
  const float tmp90 = (tmp89 * tmp87);
  const float tmp91 = (tmp47 * tmp50);
  const float tmp92 = (tmp74 * tmp88);
  const float tmp93 = (tmp77 * tmp90);
  const float tmp94 = (tmp92 + tmp93);
  const float tmp95 = (tmp80 * tmp88);
  const float tmp96 = (tmp84 * tmp90);
  const float tmp97 = (tmp95 + tmp96);
  S4 tmp4254 = tmp4183.get1(runtime_, mem_alloc_);
  S4_ch tmp4257 = tmp4254.children(tmp6408);
  device float* tmp4258 = tmp4257.get0(runtime_, mem_alloc_).val;
  const auto tmp99 = *tmp4258;
  const float tmp100 = (tmp99 * tmp72);
  const float tmp101 = (tmp100 + tmp94);
  const float tmp102 = (tmp91 * tmp101);
  device float* tmp4268 = tmp4257.get1(runtime_, mem_alloc_).val;
  const auto tmp104 = *tmp4268;
  const float tmp105 = (tmp104 * tmp72);
  const float tmp106 = (tmp105 + tmp97);
  const float tmp107 = (tmp91 * tmp106);
  S14 tmp4275 = tmp4183.get4(runtime_, mem_alloc_);
  constexpr int32_t tmp5216 = 127;
  const int32_t tmp5217 = (tmp37 & tmp5216);
  const int32_t tmp5221 = (tmp39 & tmp5216);
  constexpr int32_t tmp6429 = 7;
  const int32_t tmp6430 = (tmp5217 << tmp6429);
  const int32_t tmp5846 = (tmp5221 + tmp6430);
  S14_ch tmp4279 = tmp4275.children(tmp5846);
  device float* tmp4280 = tmp4279.get0(runtime_, mem_alloc_).val;
  const float tmp109 = fatomic_fetch_add(tmp4280, tmp102);
  device float* tmp4292 = tmp4279.get1(runtime_, mem_alloc_).val;
  const float tmp111 = fatomic_fetch_add(tmp4292, tmp107);
  const float tmp112 = (tmp91 * tmp72);
  S17 tmp4299 = tmp4183.get5(runtime_, mem_alloc_);
  S17_ch tmp4303 = tmp4299.children(tmp5846);
  device float* tmp4304 = tmp4303.get0(runtime_, mem_alloc_).val;
  const float tmp114 = fatomic_fetch_add(tmp4304, tmp112);
  const float tmp115 = (tmp51 - tmp43);
  const float tmp116 = (tmp115 * tmp87);
  const float tmp117 = (tmp47 * tmp58);
  const float tmp118 = (tmp77 * tmp116);
  const float tmp119 = (tmp92 + tmp118);
  const float tmp120 = (tmp84 * tmp116);
  const float tmp121 = (tmp95 + tmp120);
  const float tmp122 = (tmp100 + tmp119);
  const float tmp123 = (tmp117 * tmp122);
  const float tmp124 = (tmp105 + tmp121);
  const float tmp125 = (tmp117 * tmp124);
  const int32_t tmp127 = (tmp39 + tmp5796);
  const int32_t tmp5245 = (tmp127 & tmp5216);
  const int32_t tmp5870 = (tmp5245 + tmp6430);
  S14_ch tmp4315 = tmp4275.children(tmp5870);
  device float* tmp4316 = tmp4315.get0(runtime_, mem_alloc_).val;
  const float tmp129 = fatomic_fetch_add(tmp4316, tmp123);
  device float* tmp4328 = tmp4315.get1(runtime_, mem_alloc_).val;
  const float tmp131 = fatomic_fetch_add(tmp4328, tmp125);
  const float tmp132 = (tmp117 * tmp72);
  S17_ch tmp4339 = tmp4299.children(tmp5870);
  device float* tmp4340 = tmp4339.get0(runtime_, mem_alloc_).val;
  const float tmp134 = fatomic_fetch_add(tmp4340, tmp132);
  constexpr float tmp135 = 2.0;
  const float tmp136 = (tmp135 - tmp43);
  const float tmp137 = (tmp136 * tmp87);
  const float tmp138 = (tmp47 * tmp64);
  const float tmp139 = (tmp77 * tmp137);
  const float tmp140 = (tmp92 + tmp139);
  const float tmp141 = (tmp84 * tmp137);
  const float tmp142 = (tmp95 + tmp141);
  const float tmp143 = (tmp100 + tmp140);
  const float tmp144 = (tmp138 * tmp143);
  const float tmp145 = (tmp105 + tmp142);
  const float tmp146 = (tmp138 * tmp145);
  constexpr int32_t tmp147 = 2;
  const int32_t tmp148 = (tmp39 + tmp147);
  const int32_t tmp5269 = (tmp148 & tmp5216);
  const int32_t tmp5894 = (tmp5269 + tmp6430);
  S14_ch tmp4351 = tmp4275.children(tmp5894);
  device float* tmp4352 = tmp4351.get0(runtime_, mem_alloc_).val;
  const float tmp150 = fatomic_fetch_add(tmp4352, tmp144);
  device float* tmp4364 = tmp4351.get1(runtime_, mem_alloc_).val;
  const float tmp152 = fatomic_fetch_add(tmp4364, tmp146);
  const float tmp153 = (tmp138 * tmp72);
  S17_ch tmp4375 = tmp4299.children(tmp5894);
  device float* tmp4376 = tmp4375.get0(runtime_, mem_alloc_).val;
  const float tmp155 = fatomic_fetch_add(tmp4376, tmp153);
  const float tmp156 = (tmp51 - tmp41);
  const float tmp157 = (tmp156 * tmp87);
  const float tmp158 = (tmp55 * tmp50);
  const float tmp159 = (tmp74 * tmp157);
  const float tmp160 = (tmp159 + tmp93);
  const float tmp161 = (tmp80 * tmp157);
  const float tmp162 = (tmp161 + tmp96);
  const float tmp163 = (tmp100 + tmp160);
  const float tmp164 = (tmp158 * tmp163);
  const float tmp165 = (tmp105 + tmp162);
  const float tmp166 = (tmp158 * tmp165);
  const int32_t tmp167 = (tmp37 + tmp5796);
  const int32_t tmp5289 = (tmp167 & tmp5216);
  const int32_t tmp6432 = (tmp5289 << tmp6429);
  const int32_t tmp5918 = (tmp5221 + tmp6432);
  S14_ch tmp4387 = tmp4275.children(tmp5918);
  device float* tmp4388 = tmp4387.get0(runtime_, mem_alloc_).val;
  const float tmp169 = fatomic_fetch_add(tmp4388, tmp164);
  device float* tmp4400 = tmp4387.get1(runtime_, mem_alloc_).val;
  const float tmp171 = fatomic_fetch_add(tmp4400, tmp166);
  const float tmp172 = (tmp158 * tmp72);
  S17_ch tmp4411 = tmp4299.children(tmp5918);
  device float* tmp4412 = tmp4411.get0(runtime_, mem_alloc_).val;
  const float tmp174 = fatomic_fetch_add(tmp4412, tmp172);
  const float tmp175 = (tmp55 * tmp58);
  const float tmp176 = (tmp159 + tmp118);
  const float tmp177 = (tmp161 + tmp120);
  const float tmp178 = (tmp100 + tmp176);
  const float tmp179 = (tmp175 * tmp178);
  const float tmp180 = (tmp105 + tmp177);
  const float tmp181 = (tmp175 * tmp180);
  const int32_t tmp5942 = (tmp5245 + tmp6432);
  S14_ch tmp4423 = tmp4275.children(tmp5942);
  device float* tmp4424 = tmp4423.get0(runtime_, mem_alloc_).val;
  const float tmp183 = fatomic_fetch_add(tmp4424, tmp179);
  device float* tmp4436 = tmp4423.get1(runtime_, mem_alloc_).val;
  const float tmp185 = fatomic_fetch_add(tmp4436, tmp181);
  const float tmp186 = (tmp175 * tmp72);
  S17_ch tmp4447 = tmp4299.children(tmp5942);
  device float* tmp4448 = tmp4447.get0(runtime_, mem_alloc_).val;
  const float tmp188 = fatomic_fetch_add(tmp4448, tmp186);
  const float tmp189 = (tmp55 * tmp64);
  const float tmp190 = (tmp159 + tmp139);
  const float tmp191 = (tmp161 + tmp141);
  const float tmp192 = (tmp100 + tmp190);
  const float tmp193 = (tmp189 * tmp192);
  const float tmp194 = (tmp105 + tmp191);
  const float tmp195 = (tmp189 * tmp194);
  const int32_t tmp5966 = (tmp5269 + tmp6432);
  S14_ch tmp4459 = tmp4275.children(tmp5966);
  device float* tmp4460 = tmp4459.get0(runtime_, mem_alloc_).val;
  const float tmp197 = fatomic_fetch_add(tmp4460, tmp193);
  device float* tmp4472 = tmp4459.get1(runtime_, mem_alloc_).val;
  const float tmp199 = fatomic_fetch_add(tmp4472, tmp195);
  const float tmp200 = (tmp189 * tmp72);
  S17_ch tmp4483 = tmp4299.children(tmp5966);
  device float* tmp4484 = tmp4483.get0(runtime_, mem_alloc_).val;
  const float tmp202 = fatomic_fetch_add(tmp4484, tmp200);
  const float tmp203 = (tmp135 - tmp41);
  const float tmp204 = (tmp203 * tmp87);
  const float tmp205 = (tmp61 * tmp50);
  const float tmp206 = (tmp74 * tmp204);
  const float tmp207 = (tmp206 + tmp93);
  const float tmp208 = (tmp80 * tmp204);
  const float tmp209 = (tmp208 + tmp96);
  const float tmp210 = (tmp100 + tmp207);
  const float tmp211 = (tmp205 * tmp210);
  const float tmp212 = (tmp105 + tmp209);
  const float tmp213 = (tmp205 * tmp212);
  const int32_t tmp214 = (tmp37 + tmp147);
  const int32_t tmp5361 = (tmp214 & tmp5216);
  const int32_t tmp6434 = (tmp5361 << tmp6429);
  const int32_t tmp5990 = (tmp5221 + tmp6434);
  S14_ch tmp4495 = tmp4275.children(tmp5990);
  device float* tmp4496 = tmp4495.get0(runtime_, mem_alloc_).val;
  const float tmp216 = fatomic_fetch_add(tmp4496, tmp211);
  device float* tmp4508 = tmp4495.get1(runtime_, mem_alloc_).val;
  const float tmp218 = fatomic_fetch_add(tmp4508, tmp213);
  const float tmp219 = (tmp205 * tmp72);
  S17_ch tmp4519 = tmp4299.children(tmp5990);
  device float* tmp4520 = tmp4519.get0(runtime_, mem_alloc_).val;
  const float tmp221 = fatomic_fetch_add(tmp4520, tmp219);
  const float tmp222 = (tmp61 * tmp58);
  const float tmp223 = (tmp206 + tmp118);
  const float tmp224 = (tmp208 + tmp120);
  const float tmp225 = (tmp100 + tmp223);
  const float tmp226 = (tmp222 * tmp225);
  const float tmp227 = (tmp105 + tmp224);
  const float tmp228 = (tmp222 * tmp227);
  const int32_t tmp6014 = (tmp5245 + tmp6434);
  S14_ch tmp4531 = tmp4275.children(tmp6014);
  device float* tmp4532 = tmp4531.get0(runtime_, mem_alloc_).val;
  const float tmp230 = fatomic_fetch_add(tmp4532, tmp226);
  device float* tmp4544 = tmp4531.get1(runtime_, mem_alloc_).val;
  const float tmp232 = fatomic_fetch_add(tmp4544, tmp228);
  const float tmp233 = (tmp222 * tmp72);
  S17_ch tmp4555 = tmp4299.children(tmp6014);
  device float* tmp4556 = tmp4555.get0(runtime_, mem_alloc_).val;
  const float tmp235 = fatomic_fetch_add(tmp4556, tmp233);
  const float tmp236 = (tmp61 * tmp64);
  const float tmp237 = (tmp206 + tmp139);
  const float tmp238 = (tmp208 + tmp141);
  const float tmp239 = (tmp100 + tmp237);
  const float tmp240 = (tmp236 * tmp239);
  const float tmp241 = (tmp105 + tmp238);
  const float tmp242 = (tmp236 * tmp241);
  const int32_t tmp6038 = (tmp5269 + tmp6434);
  S14_ch tmp4567 = tmp4275.children(tmp6038);
  device float* tmp4568 = tmp4567.get0(runtime_, mem_alloc_).val;
  const float tmp244 = fatomic_fetch_add(tmp4568, tmp240);
  device float* tmp4580 = tmp4567.get1(runtime_, mem_alloc_).val;
  const float tmp246 = fatomic_fetch_add(tmp4580, tmp242);
  const float tmp247 = (tmp236 * tmp72);
  S17_ch tmp4591 = tmp4299.children(tmp6038);
  device float* tmp4592 = tmp4591.get0(runtime_, mem_alloc_).val;
  const float tmp249 = fatomic_fetch_add(tmp4592, tmp247);
}

void mtl_k0004_substep_c4_0_2_func(
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
  constexpr int32_t tmp5564 = 127;
  const int tmp253 = linear_loop_idx_;
  constexpr int32_t tmp5430 = 7;
  const int32_t tmp5431 = (tmp253 >> tmp5430);
  const int32_t tmp5433 = (tmp5431 & tmp5564);
  const int32_t tmp5437 = (tmp253 & tmp5564);
  constexpr float tmp263 = 0.0;
  S0 tmp4596(root_addr);
  constexpr int32_t tmp6055 = 0;
  S0_ch tmp4598 = tmp4596.children(tmp6055);
  S17 tmp4599 = tmp4598.get5(runtime_, mem_alloc_);
  constexpr int32_t tmp6057 = 1;
  const int32_t tmp6436 = (tmp5433 << tmp5430);
  const int32_t tmp6062 = (tmp5437 + tmp6436);
  S17_ch tmp4603 = tmp4599.children(tmp6062);
  device float* tmp4604 = tmp4603.get0(runtime_, mem_alloc_).val;
  const auto tmp265 = *tmp4604;
  const int32_t tmp266 = -(tmp265 > tmp263);
  const int32_t tmp268 = (tmp266 & tmp6057);
  if (tmp268) {
    S14 tmp4611 = tmp4598.get4(runtime_, mem_alloc_);
    S14_ch tmp4615 = tmp4611.children(tmp6062);
    device float* tmp4616 = tmp4615.get0(runtime_, mem_alloc_).val;
    const auto tmp271 = *tmp4616;
    const auto tmp272 = *tmp4604;
    const float tmp273 = (tmp271 / tmp272);
    device float* tmp4640 = tmp4615.get1(runtime_, mem_alloc_).val;
    const auto tmp275 = *tmp4640;
    const float tmp276 = (tmp275 / tmp272);
    *tmp4616 = tmp273;
    *tmp4640 = tmp276;
  } else {
  }
  constexpr float tmp280 = -0.00196;
  S14 tmp4671 = tmp4598.get4(runtime_, mem_alloc_);
  S14_ch tmp4675 = tmp4671.children(tmp6062);
  device float* tmp4676 = tmp4675.get1(runtime_, mem_alloc_).val;
  const auto tmp281 = *tmp4676;
  const float tmp282 = (tmp281 + tmp280);
  *tmp4676 = tmp282;
  constexpr int32_t tmp284 = 3;
  const int32_t tmp285 = -(tmp5433 < tmp284);
  const int32_t tmp286 = (tmp285 & tmp6057);
  device float* tmp4700 = tmp4675.get0(runtime_, mem_alloc_).val;
  const auto tmp288 = *tmp4700;
  const int32_t tmp289 = -(tmp288 < tmp263);
  const int32_t tmp290 = (tmp289 & tmp6057);
  const int32_t tmp291 = (tmp286 & tmp290);
  if (tmp291) {
    *tmp4700 = tmp263;
  } else {
  }
  constexpr int32_t tmp294 = 125;
  const int32_t tmp295 = -(tmp5433 > tmp294);
  const int32_t tmp296 = (tmp295 & tmp6057);
  const auto tmp297 = *tmp4700;
  const int32_t tmp298 = -(tmp297 > tmp263);
  const int32_t tmp299 = (tmp298 & tmp6057);
  const int32_t tmp300 = (tmp296 & tmp299);
  if (tmp300) {
    *tmp4700 = tmp263;
  } else {
  }
  const int32_t tmp303 = -(tmp5437 < tmp284);
  const int32_t tmp304 = (tmp303 & tmp6057);
  const auto tmp305 = *tmp4676;
  const int32_t tmp306 = -(tmp305 < tmp263);
  const int32_t tmp307 = (tmp306 & tmp6057);
  const int32_t tmp308 = (tmp304 & tmp307);
  if (tmp308) {
    *tmp4676 = tmp263;
  } else {
  }
  const int32_t tmp311 = -(tmp5437 > tmp294);
  const int32_t tmp312 = (tmp311 & tmp6057);
  const auto tmp313 = *tmp4676;
  const int32_t tmp314 = -(tmp313 > tmp263);
  const int32_t tmp315 = (tmp314 & tmp6057);
  const int32_t tmp316 = (tmp312 & tmp315);
  if (tmp316) {
    *tmp4676 = tmp263;
  } else {
  }
}

void mtl_k0004_substep_c4_0_3_func(
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
  const int tmp321 = linear_loop_idx_;
  S0 tmp4787(root_addr);
  constexpr int32_t tmp6183 = 0;
  S0_ch tmp4789 = tmp4787.children(tmp6183);
  S1 tmp4790 = tmp4789.get0(runtime_, mem_alloc_);
  constexpr int32_t tmp6426 = 8191;
  const int32_t tmp6414 = (tmp321 & tmp6426);
  constexpr int32_t tmp6185 = 1;
  S1_ch tmp4793 = tmp4790.children(tmp6414);
  device float* tmp4794 = tmp4793.get0(runtime_, mem_alloc_).val;
  const auto tmp328 = *tmp4794;
  constexpr float tmp329 = 128.0;
  const float tmp330 = (tmp328 * tmp329);
  device float* tmp4804 = tmp4793.get1(runtime_, mem_alloc_).val;
  const auto tmp332 = *tmp4804;
  const float tmp333 = (tmp332 * tmp329);
  constexpr float tmp334 = 0.5;
  const float tmp335 = (tmp330 - tmp334);
  const int32_t tmp336 = static_cast<int32_t>(tmp335);
  const float tmp337 = (tmp333 - tmp334);
  const int32_t tmp338 = static_cast<int32_t>(tmp337);
  const float tmp339 = static_cast<float>(tmp336);
  const float tmp340 = (tmp330 - tmp339);
  const float tmp341 = static_cast<float>(tmp338);
  const float tmp342 = (tmp333 - tmp341);
  constexpr float tmp343 = 1.5;
  const float tmp344 = (tmp343 - tmp340);
  const float tmp345 = (tmp344 * tmp344);
  const float tmp346 = (tmp345 * tmp334);
  const float tmp347 = (tmp343 - tmp342);
  const float tmp348 = (tmp347 * tmp347);
  const float tmp349 = (tmp348 * tmp334);
  constexpr float tmp350 = 1.0;
  const float tmp351 = (tmp340 - tmp350);
  const float tmp352 = (tmp351 * tmp351);
  constexpr float tmp353 = 0.75;
  const float tmp354 = (tmp353 - tmp352);
  const float tmp355 = (tmp342 - tmp350);
  const float tmp356 = (tmp355 * tmp355);
  const float tmp357 = (tmp353 - tmp356);
  const float tmp358 = (tmp340 - tmp334);
  const float tmp359 = (tmp358 * tmp358);
  const float tmp360 = (tmp359 * tmp334);
  const float tmp361 = (tmp342 - tmp334);
  const float tmp362 = (tmp361 * tmp361);
  const float tmp363 = (tmp362 * tmp334);
  constexpr float tmp370 = 0.0;
  const float tmp371 = (tmp370 - tmp340);
  constexpr float tmp372 = 0.0078125;
  const float tmp373 = (tmp371 * tmp372);
  const float tmp374 = (tmp370 - tmp342);
  const float tmp375 = (tmp374 * tmp372);
  const float tmp376 = (tmp346 * tmp349);
  S14 tmp4811 = tmp4789.get4(runtime_, mem_alloc_);
  constexpr int32_t tmp5580 = 127;
  const int32_t tmp5581 = (tmp336 & tmp5580);
  const int32_t tmp5585 = (tmp338 & tmp5580);
  constexpr int32_t tmp6437 = 7;
  const int32_t tmp6438 = (tmp5581 << tmp6437);
  const int32_t tmp6200 = (tmp5585 + tmp6438);
  S14_ch tmp4815 = tmp4811.children(tmp6200);
  device float* tmp4816 = tmp4815.get0(runtime_, mem_alloc_).val;
  const auto tmp378 = *tmp4816;
  device float* tmp4828 = tmp4815.get1(runtime_, mem_alloc_).val;
  const auto tmp380 = *tmp4828;
  const float tmp381 = (tmp376 * tmp378);
  const float tmp382 = (tmp376 * tmp380);
  const float tmp389 = (tmp378 * tmp373);
  const float tmp390 = (tmp378 * tmp375);
  const float tmp391 = (tmp380 * tmp373);
  const float tmp392 = (tmp380 * tmp375);
  constexpr float tmp393 = 4.0;
  const float tmp394 = (tmp376 * tmp393);
  const float tmp395 = (tmp394 * tmp389);
  constexpr float tmp396 = 16384.0;
  const float tmp397 = (tmp395 * tmp396);
  const float tmp398 = (tmp394 * tmp390);
  const float tmp399 = (tmp398 * tmp396);
  const float tmp400 = (tmp394 * tmp391);
  const float tmp401 = (tmp400 * tmp396);
  const float tmp402 = (tmp394 * tmp392);
  const float tmp403 = (tmp402 * tmp396);
  const float tmp416 = (tmp350 - tmp342);
  const float tmp417 = (tmp416 * tmp372);
  const float tmp418 = (tmp346 * tmp357);
  const int32_t tmp420 = (tmp338 + tmp6185);
  const int32_t tmp5601 = (tmp420 & tmp5580);
  const int32_t tmp6216 = (tmp5601 + tmp6438);
  S14_ch tmp4839 = tmp4811.children(tmp6216);
  device float* tmp4840 = tmp4839.get0(runtime_, mem_alloc_).val;
  const auto tmp422 = *tmp4840;
  device float* tmp4852 = tmp4839.get1(runtime_, mem_alloc_).val;
  const auto tmp424 = *tmp4852;
  const float tmp425 = (tmp418 * tmp422);
  const float tmp426 = (tmp418 * tmp424);
  const float tmp428 = (tmp381 + tmp425);
  const float tmp431 = (tmp382 + tmp426);
  const float tmp433 = (tmp422 * tmp373);
  const float tmp434 = (tmp422 * tmp417);
  const float tmp435 = (tmp424 * tmp373);
  const float tmp436 = (tmp424 * tmp417);
  const float tmp437 = (tmp418 * tmp393);
  const float tmp438 = (tmp437 * tmp433);
  const float tmp439 = (tmp438 * tmp396);
  const float tmp440 = (tmp437 * tmp434);
  const float tmp441 = (tmp440 * tmp396);
  const float tmp442 = (tmp437 * tmp435);
  const float tmp443 = (tmp442 * tmp396);
  const float tmp444 = (tmp437 * tmp436);
  const float tmp445 = (tmp444 * tmp396);
  const float tmp447 = (tmp397 + tmp439);
  const float tmp450 = (tmp399 + tmp441);
  const float tmp453 = (tmp401 + tmp443);
  const float tmp456 = (tmp403 + tmp445);
  constexpr float tmp458 = 2.0;
  const float tmp459 = (tmp458 - tmp342);
  const float tmp460 = (tmp459 * tmp372);
  const float tmp461 = (tmp346 * tmp363);
  constexpr int32_t tmp462 = 2;
  const int32_t tmp463 = (tmp338 + tmp462);
  const int32_t tmp5617 = (tmp463 & tmp5580);
  const int32_t tmp6232 = (tmp5617 + tmp6438);
  S14_ch tmp4863 = tmp4811.children(tmp6232);
  device float* tmp4864 = tmp4863.get0(runtime_, mem_alloc_).val;
  const auto tmp465 = *tmp4864;
  device float* tmp4876 = tmp4863.get1(runtime_, mem_alloc_).val;
  const auto tmp467 = *tmp4876;
  const float tmp468 = (tmp461 * tmp465);
  const float tmp469 = (tmp461 * tmp467);
  const float tmp471 = (tmp428 + tmp468);
  const float tmp474 = (tmp431 + tmp469);
  const float tmp476 = (tmp465 * tmp373);
  const float tmp477 = (tmp465 * tmp460);
  const float tmp478 = (tmp467 * tmp373);
  const float tmp479 = (tmp467 * tmp460);
  const float tmp480 = (tmp461 * tmp393);
  const float tmp481 = (tmp480 * tmp476);
  const float tmp482 = (tmp481 * tmp396);
  const float tmp483 = (tmp480 * tmp477);
  const float tmp484 = (tmp483 * tmp396);
  const float tmp485 = (tmp480 * tmp478);
  const float tmp486 = (tmp485 * tmp396);
  const float tmp487 = (tmp480 * tmp479);
  const float tmp488 = (tmp487 * tmp396);
  const float tmp490 = (tmp447 + tmp482);
  const float tmp493 = (tmp450 + tmp484);
  const float tmp496 = (tmp453 + tmp486);
  const float tmp499 = (tmp456 + tmp488);
  const float tmp501 = (tmp350 - tmp340);
  const float tmp502 = (tmp501 * tmp372);
  const float tmp503 = (tmp354 * tmp349);
  const int32_t tmp504 = (tmp336 + tmp6185);
  const int32_t tmp5629 = (tmp504 & tmp5580);
  const int32_t tmp6440 = (tmp5629 << tmp6437);
  const int32_t tmp6248 = (tmp5585 + tmp6440);
  S14_ch tmp4887 = tmp4811.children(tmp6248);
  device float* tmp4888 = tmp4887.get0(runtime_, mem_alloc_).val;
  const auto tmp506 = *tmp4888;
  device float* tmp4900 = tmp4887.get1(runtime_, mem_alloc_).val;
  const auto tmp508 = *tmp4900;
  const float tmp509 = (tmp503 * tmp506);
  const float tmp510 = (tmp503 * tmp508);
  const float tmp512 = (tmp471 + tmp509);
  const float tmp515 = (tmp474 + tmp510);
  const float tmp517 = (tmp506 * tmp502);
  const float tmp518 = (tmp506 * tmp375);
  const float tmp519 = (tmp508 * tmp502);
  const float tmp520 = (tmp508 * tmp375);
  const float tmp521 = (tmp503 * tmp393);
  const float tmp522 = (tmp521 * tmp517);
  const float tmp523 = (tmp522 * tmp396);
  const float tmp524 = (tmp521 * tmp518);
  const float tmp525 = (tmp524 * tmp396);
  const float tmp526 = (tmp521 * tmp519);
  const float tmp527 = (tmp526 * tmp396);
  const float tmp528 = (tmp521 * tmp520);
  const float tmp529 = (tmp528 * tmp396);
  const float tmp531 = (tmp490 + tmp523);
  const float tmp534 = (tmp493 + tmp525);
  const float tmp537 = (tmp496 + tmp527);
  const float tmp540 = (tmp499 + tmp529);
  const float tmp542 = (tmp354 * tmp357);
  const int32_t tmp6264 = (tmp5601 + tmp6440);
  S14_ch tmp4911 = tmp4811.children(tmp6264);
  device float* tmp4912 = tmp4911.get0(runtime_, mem_alloc_).val;
  const auto tmp544 = *tmp4912;
  device float* tmp4924 = tmp4911.get1(runtime_, mem_alloc_).val;
  const auto tmp546 = *tmp4924;
  const float tmp547 = (tmp542 * tmp544);
  const float tmp548 = (tmp542 * tmp546);
  const float tmp550 = (tmp512 + tmp547);
  const float tmp553 = (tmp515 + tmp548);
  const float tmp555 = (tmp544 * tmp502);
  const float tmp556 = (tmp544 * tmp417);
  const float tmp557 = (tmp546 * tmp502);
  const float tmp558 = (tmp546 * tmp417);
  const float tmp559 = (tmp542 * tmp393);
  const float tmp560 = (tmp559 * tmp555);
  const float tmp561 = (tmp560 * tmp396);
  const float tmp562 = (tmp559 * tmp556);
  const float tmp563 = (tmp562 * tmp396);
  const float tmp564 = (tmp559 * tmp557);
  const float tmp565 = (tmp564 * tmp396);
  const float tmp566 = (tmp559 * tmp558);
  const float tmp567 = (tmp566 * tmp396);
  const float tmp569 = (tmp531 + tmp561);
  const float tmp572 = (tmp534 + tmp563);
  const float tmp575 = (tmp537 + tmp565);
  const float tmp578 = (tmp540 + tmp567);
  const float tmp580 = (tmp354 * tmp363);
  const int32_t tmp6280 = (tmp5617 + tmp6440);
  S14_ch tmp4935 = tmp4811.children(tmp6280);
  device float* tmp4936 = tmp4935.get0(runtime_, mem_alloc_).val;
  const auto tmp582 = *tmp4936;
  device float* tmp4948 = tmp4935.get1(runtime_, mem_alloc_).val;
  const auto tmp584 = *tmp4948;
  const float tmp585 = (tmp580 * tmp582);
  const float tmp586 = (tmp580 * tmp584);
  const float tmp588 = (tmp550 + tmp585);
  const float tmp591 = (tmp553 + tmp586);
  const float tmp593 = (tmp582 * tmp502);
  const float tmp594 = (tmp582 * tmp460);
  const float tmp595 = (tmp584 * tmp502);
  const float tmp596 = (tmp584 * tmp460);
  const float tmp597 = (tmp580 * tmp393);
  const float tmp598 = (tmp597 * tmp593);
  const float tmp599 = (tmp598 * tmp396);
  const float tmp600 = (tmp597 * tmp594);
  const float tmp601 = (tmp600 * tmp396);
  const float tmp602 = (tmp597 * tmp595);
  const float tmp603 = (tmp602 * tmp396);
  const float tmp604 = (tmp597 * tmp596);
  const float tmp605 = (tmp604 * tmp396);
  const float tmp607 = (tmp569 + tmp599);
  const float tmp610 = (tmp572 + tmp601);
  const float tmp613 = (tmp575 + tmp603);
  const float tmp616 = (tmp578 + tmp605);
  const float tmp618 = (tmp458 - tmp340);
  const float tmp619 = (tmp618 * tmp372);
  const float tmp620 = (tmp360 * tmp349);
  const int32_t tmp621 = (tmp336 + tmp462);
  const int32_t tmp5677 = (tmp621 & tmp5580);
  const int32_t tmp6442 = (tmp5677 << tmp6437);
  const int32_t tmp6296 = (tmp5585 + tmp6442);
  S14_ch tmp4959 = tmp4811.children(tmp6296);
  device float* tmp4960 = tmp4959.get0(runtime_, mem_alloc_).val;
  const auto tmp623 = *tmp4960;
  device float* tmp4972 = tmp4959.get1(runtime_, mem_alloc_).val;
  const auto tmp625 = *tmp4972;
  const float tmp626 = (tmp620 * tmp623);
  const float tmp627 = (tmp620 * tmp625);
  const float tmp629 = (tmp588 + tmp626);
  const float tmp632 = (tmp591 + tmp627);
  const float tmp634 = (tmp623 * tmp619);
  const float tmp635 = (tmp623 * tmp375);
  const float tmp636 = (tmp625 * tmp619);
  const float tmp637 = (tmp625 * tmp375);
  const float tmp638 = (tmp620 * tmp393);
  const float tmp639 = (tmp638 * tmp634);
  const float tmp640 = (tmp639 * tmp396);
  const float tmp641 = (tmp638 * tmp635);
  const float tmp642 = (tmp641 * tmp396);
  const float tmp643 = (tmp638 * tmp636);
  const float tmp644 = (tmp643 * tmp396);
  const float tmp645 = (tmp638 * tmp637);
  const float tmp646 = (tmp645 * tmp396);
  const float tmp648 = (tmp607 + tmp640);
  const float tmp651 = (tmp610 + tmp642);
  const float tmp654 = (tmp613 + tmp644);
  const float tmp657 = (tmp616 + tmp646);
  const float tmp659 = (tmp360 * tmp357);
  const int32_t tmp6312 = (tmp5601 + tmp6442);
  S14_ch tmp4983 = tmp4811.children(tmp6312);
  device float* tmp4984 = tmp4983.get0(runtime_, mem_alloc_).val;
  const auto tmp661 = *tmp4984;
  device float* tmp4996 = tmp4983.get1(runtime_, mem_alloc_).val;
  const auto tmp663 = *tmp4996;
  const float tmp664 = (tmp659 * tmp661);
  const float tmp665 = (tmp659 * tmp663);
  const float tmp667 = (tmp629 + tmp664);
  const float tmp670 = (tmp632 + tmp665);
  const float tmp672 = (tmp661 * tmp619);
  const float tmp673 = (tmp661 * tmp417);
  const float tmp674 = (tmp663 * tmp619);
  const float tmp675 = (tmp663 * tmp417);
  const float tmp676 = (tmp659 * tmp393);
  const float tmp677 = (tmp676 * tmp672);
  const float tmp678 = (tmp677 * tmp396);
  const float tmp679 = (tmp676 * tmp673);
  const float tmp680 = (tmp679 * tmp396);
  const float tmp681 = (tmp676 * tmp674);
  const float tmp682 = (tmp681 * tmp396);
  const float tmp683 = (tmp676 * tmp675);
  const float tmp684 = (tmp683 * tmp396);
  const float tmp686 = (tmp648 + tmp678);
  const float tmp689 = (tmp651 + tmp680);
  const float tmp692 = (tmp654 + tmp682);
  const float tmp695 = (tmp657 + tmp684);
  const float tmp697 = (tmp360 * tmp363);
  const int32_t tmp6328 = (tmp5617 + tmp6442);
  S14_ch tmp5007 = tmp4811.children(tmp6328);
  device float* tmp5008 = tmp5007.get0(runtime_, mem_alloc_).val;
  const auto tmp699 = *tmp5008;
  device float* tmp5020 = tmp5007.get1(runtime_, mem_alloc_).val;
  const auto tmp701 = *tmp5020;
  const float tmp702 = (tmp697 * tmp699);
  const float tmp703 = (tmp697 * tmp701);
  const float tmp705 = (tmp667 + tmp702);
  const float tmp708 = (tmp670 + tmp703);
  const float tmp710 = (tmp699 * tmp619);
  const float tmp711 = (tmp699 * tmp460);
  const float tmp712 = (tmp701 * tmp619);
  const float tmp713 = (tmp701 * tmp460);
  const float tmp714 = (tmp697 * tmp393);
  const float tmp715 = (tmp714 * tmp710);
  const float tmp716 = (tmp715 * tmp396);
  const float tmp717 = (tmp714 * tmp711);
  const float tmp718 = (tmp717 * tmp396);
  const float tmp719 = (tmp714 * tmp712);
  const float tmp720 = (tmp719 * tmp396);
  const float tmp721 = (tmp714 * tmp713);
  const float tmp722 = (tmp721 * tmp396);
  const float tmp724 = (tmp686 + tmp716);
  const float tmp727 = (tmp689 + tmp718);
  const float tmp730 = (tmp692 + tmp720);
  const float tmp733 = (tmp695 + tmp722);
  S4 tmp5026 = tmp4789.get1(runtime_, mem_alloc_);
  S4_ch tmp5029 = tmp5026.children(tmp6414);
  device float* tmp5030 = tmp5029.get0(runtime_, mem_alloc_).val;
  *tmp5030 = tmp705;
  device float* tmp5040 = tmp5029.get1(runtime_, mem_alloc_).val;
  *tmp5040 = tmp708;
  constexpr float tmp741 = 0.0002;
  const float tmp742 = (tmp705 * tmp741);
  const float tmp743 = (tmp708 * tmp741);
  const float tmp745 = (tmp328 + tmp742);
  *tmp4794 = tmp745;
  const float tmp748 = (tmp332 + tmp743);
  *tmp4804 = tmp748;
  S12 tmp5086 = tmp4789.get3(runtime_, mem_alloc_);
  S12_ch tmp5089 = tmp5086.children(tmp6414);
  device float* tmp5090 = tmp5089.get0(runtime_, mem_alloc_).val;
  const auto tmp751 = *tmp5090;
  const float tmp754 = (tmp724 + tmp733);
  const float tmp755 = (tmp754 * tmp741);
  const float tmp756 = (tmp755 + tmp350);
  const float tmp757 = (tmp751 * tmp756);
  *tmp5090 = tmp757;
  S7 tmp5106 = tmp4789.get2(runtime_, mem_alloc_);
  S7_ch tmp5109 = tmp5106.children(tmp6414);
  device float* tmp5110 = tmp5109.get0(runtime_, mem_alloc_).val;
  *tmp5110 = tmp724;
  device float* tmp5120 = tmp5109.get1(runtime_, mem_alloc_).val;
  *tmp5120 = tmp727;
  device float* tmp5130 = tmp5109.get2(runtime_, mem_alloc_).val;
  *tmp5130 = tmp730;
  device float* tmp5140 = tmp5109.get3(runtime_, mem_alloc_).val;
  *tmp5140 = tmp733;
}

}  // namespace
kernel void mtl_k0004_substep_c4_0_0(
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
    mtl_k0004_substep_c4_0_0_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0004_substep_c4_0_1(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 8192;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0004_substep_c4_0_1_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0004_substep_c4_0_2(
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
    mtl_k0004_substep_c4_0_2_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

kernel void mtl_k0004_substep_c4_0_3(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 8192;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0004_substep_c4_0_3_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

