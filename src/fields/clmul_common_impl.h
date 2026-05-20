/**********************************************************************
 * Copyright (c) 2018 Pieter Wuille, Greg Maxwell, Gleb Naumenko      *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef _MINISKETCH_FIELDS_CLMUL_COMMON_IMPL_H_
#define _MINISKETCH_FIELDS_CLMUL_COMMON_IMPL_H_ 1

#include <stdint.h>

#if defined(__PCLMUL__)
#  include <immintrin.h>
#  define MINISKETCH_CLMUL_X86 1
#elif defined(__ARM_FEATURE_CRYPTO) || defined(__ARM_FEATURE_AES)
#  include <arm_neon.h>
#  define MINISKETCH_CLMUL_ARM 1
#else
#  error "clmul_common_impl.h included without a carryless-multiply ISA enabled"
#endif

#include "../int_utils.h"
#include "../lintrans.h"

namespace {

// The memory sanitizer in clang < 11 cannot reason through _mm_clmulepi64_si128 calls.
// Disable memory sanitization in the functions using them for those compilers.
#if defined(__clang__) && (__clang_major__ < 11)
#  if defined(__has_feature)
#    if __has_feature(memory_sanitizer)
#      define NO_SANITIZE_MEMORY __attribute__((no_sanitize("memory")))
#    endif
#  endif
#endif
#ifndef NO_SANITIZE_MEMORY
#  define NO_SANITIZE_MEMORY
#endif

// Thin portability layer over the carryless-multiply primitive. MulWithClMulReduce
// and MulTrinomial below are written against this API; adding a new ISA amounts to
// supplying a new body for each wrapper.
//
// Two type aliases are exposed:
//   Clmul128  — 128-bit in-flight data (inputs to XOR, OR, and the lane shifts).
//   ClmulPoly — the type a PMULL/PCLMULQDQ operand wants. On x86 the two aliases
//               coincide; on ARM ClmulPoly is poly64x2_t so vmull_p64 reads
//               naturally without polluting the lane-shift code with poly casts.
//               ClmulAsPoly is the zero-cost conversion from the in-flight type
//               to the operand type.
#if defined(MINISKETCH_CLMUL_X86)
using Clmul128 = __m128i;
using ClmulPoly = __m128i;

static inline ClmulPoly ClmulAsPoly(Clmul128 v) { return v; }
static inline ClmulPoly ClmulLoad64(uint64_t x) { return _mm_cvtsi64_si128(x); }
static inline uint64_t ClmulExtract64(Clmul128 v) { return _mm_cvtsi128_si64(v); }
// lo(a) × lo(b).
static inline Clmul128 Clmul00(ClmulPoly a, ClmulPoly b) { return _mm_clmulepi64_si128(a, b, 0x00); }
// hi(a) × lo(b). (imm8 bit 0 selects the high half of a; bit 4 clear selects the low half of b.)
static inline Clmul128 Clmul01(ClmulPoly a, ClmulPoly b) { return _mm_clmulepi64_si128(a, b, 0x01); }
static inline Clmul128 ClmulXor(Clmul128 a, Clmul128 b) { return _mm_xor_si128(a, b); }
static inline Clmul128 ClmulOr(Clmul128 a, Clmul128 b) { return _mm_or_si128(a, b); }
template<int N> static inline Clmul128 ClmulSrliEpi64(Clmul128 v) { return _mm_srli_epi64(v, N); }
template<int N> static inline Clmul128 ClmulSlliEpi64(Clmul128 v) { return _mm_slli_epi64(v, N); }
template<int N> static inline Clmul128 ClmulSrliBytes(Clmul128 v) { return _mm_srli_si128(v, N); }
#elif defined(MINISKETCH_CLMUL_ARM)
// In-flight data is uint64x2_t — directly consumed by vshrq_n_u64, vshlq_n_u64,
// veorq_u64, vorrq_u64. PMULL operands are poly64x2_t, the type vmull_p64's scalar
// arguments are extracted from via vgetq_lane_p64. ClmulAsPoly is a zero-cost
// reinterpret. With GCC 13+/Clang 19+ at -O2+, vgetq_lane_p64 + vmull_p64 fuse
// into a direct PMULL with NEON operands — values don't spill to GPRs.
using Clmul128 = uint64x2_t;
using ClmulPoly = poly64x2_t;

static inline ClmulPoly ClmulAsPoly(Clmul128 v) { return vreinterpretq_p64_u64(v); }
// Load a scalar into lane 0, lane 1 zeroed — mirrors _mm_cvtsi64_si128's
// "payload in the low 64 bits" semantics.
static inline ClmulPoly ClmulLoad64(uint64_t x) {
    return vsetq_lane_p64((poly64_t)x, vdupq_n_p64(0), 0);
}
static inline uint64_t ClmulExtract64(Clmul128 v) { return vgetq_lane_u64(v, 0); }
// lo(a) × lo(b) via PMULL.
static inline Clmul128 Clmul00(ClmulPoly a, ClmulPoly b) {
    return vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(a, 0), vgetq_lane_p64(b, 0)));
}
// hi(a) × lo(b) via PMULL — mirrors _mm_clmulepi64_si128(a, b, 0x01), which selects
// the high quadword of a (imm8 bit 0 set) and the low quadword of b (imm8 bit 4 clear).
static inline Clmul128 Clmul01(ClmulPoly a, ClmulPoly b) {
    return vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(a, 1), vgetq_lane_p64(b, 0)));
}
static inline Clmul128 ClmulXor(Clmul128 a, Clmul128 b) { return veorq_u64(a, b); }
static inline Clmul128 ClmulOr(Clmul128 a, Clmul128 b) { return vorrq_u64(a, b); }
template<int N> static inline Clmul128 ClmulSrliEpi64(Clmul128 v) { return vshrq_n_u64(v, N); }
template<int N> static inline Clmul128 ClmulSlliEpi64(Clmul128 v) { return vshlq_n_u64(v, N); }
template<int N> static inline Clmul128 ClmulSrliBytes(Clmul128 v) {
    return vreinterpretq_u64_u8(vextq_u8(vreinterpretq_u8_u64(v), vdupq_n_u8(0), N));
}
#endif

template<typename I, int BITS, I MOD> NO_SANITIZE_MEMORY I MulWithClMulReduce(I a, I b)
{
    static constexpr I MASK = Mask<BITS, I>();

    const ClmulPoly MOD128 = ClmulLoad64(MOD);
    Clmul128 product = Clmul00(ClmulLoad64((uint64_t)a), ClmulLoad64((uint64_t)b));
    if (BITS <= 32) {
        Clmul128 high1 = ClmulSrliEpi64<BITS>(product);
        Clmul128 red1 = Clmul00(ClmulAsPoly(high1), MOD128);
        Clmul128 high2 = ClmulSrliEpi64<BITS>(red1);
        Clmul128 red2 = Clmul00(ClmulAsPoly(high2), MOD128);
        return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
    } else if (BITS == 64) {
        Clmul128 red1 = Clmul01(ClmulAsPoly(product), MOD128);
        Clmul128 red2 = Clmul01(ClmulAsPoly(red1), MOD128);
        return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2));
    } else if ((BITS % 8) == 0) {
        Clmul128 high1 = ClmulSrliBytes<BITS / 8>(product);
        Clmul128 red1 = Clmul00(ClmulAsPoly(high1), MOD128);
        Clmul128 high2 = ClmulSrliBytes<BITS / 8>(red1);
        Clmul128 red2 = Clmul00(ClmulAsPoly(high2), MOD128);
        return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
    } else {
        Clmul128 high1 = ClmulOr(ClmulSrliEpi64<BITS>(product), ClmulSrliBytes<8>(ClmulSlliEpi64<64 - BITS>(product)));
        Clmul128 red1 = Clmul00(ClmulAsPoly(high1), MOD128);
        if ((uint64_t(MOD) >> (66 - BITS)) == 0) {
            Clmul128 high2 = ClmulSrliEpi64<BITS>(red1);
            Clmul128 red2 = Clmul00(ClmulAsPoly(high2), MOD128);
            return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
        } else {
            Clmul128 high2 = ClmulOr(ClmulSrliEpi64<BITS>(red1), ClmulSrliBytes<8>(ClmulSlliEpi64<64 - BITS>(red1)));
            Clmul128 red2 = Clmul00(ClmulAsPoly(high2), MOD128);
            return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
        }
    }
}

template<typename I, int BITS, int POS> NO_SANITIZE_MEMORY I MulTrinomial(I a, I b)
{
    static constexpr I MASK = Mask<BITS, I>();

    Clmul128 product = Clmul00(ClmulLoad64((uint64_t)a), ClmulLoad64((uint64_t)b));
    if (BITS <= 32) {
        Clmul128 high1 = ClmulSrliEpi64<BITS>(product);
        Clmul128 red1 = ClmulXor(high1, ClmulSlliEpi64<POS>(high1));
        if (POS == 1) {
            return ClmulExtract64(ClmulXor(product, red1)) & MASK;
        } else {
            Clmul128 high2 = ClmulSrliEpi64<BITS>(red1);
            Clmul128 red2 = ClmulXor(high2, ClmulSlliEpi64<POS>(high2));
            return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
        }
    } else {
        Clmul128 high1 = ClmulOr(ClmulSrliEpi64<BITS>(product), ClmulSrliBytes<8>(ClmulSlliEpi64<64 - BITS>(product)));
        if (BITS + POS <= 66) {
            Clmul128 red1 = ClmulXor(high1, ClmulSlliEpi64<POS>(high1));
            if (POS == 1) {
                return ClmulExtract64(ClmulXor(product, red1)) & MASK;
            } else if (BITS + POS <= 66) {
                Clmul128 high2 = ClmulSrliEpi64<BITS>(red1);
                Clmul128 red2 = ClmulXor(high2, ClmulSlliEpi64<POS>(high2));
                return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
            }
        } else {
            const ClmulPoly MOD128 = ClmulLoad64(1 + (((uint64_t)1) << POS));
            Clmul128 red1 = Clmul00(ClmulAsPoly(high1), MOD128);
            Clmul128 high2 = ClmulOr(ClmulSrliEpi64<BITS>(red1), ClmulSrliBytes<8>(ClmulSlliEpi64<64 - BITS>(red1)));
            Clmul128 red2 = ClmulXor(high2, ClmulSlliEpi64<POS>(high2));
            return ClmulExtract64(ClmulXor(ClmulXor(product, red1), red2)) & MASK;
        }
    }
}

/** Implementation of fields that use the SSE clmul intrinsic for multiplication. */
template<typename I, int B, I MOD, I (*MUL)(I, I), typename F, const F* SQR, const F* SQR2, const F* SQR4, const F* SQR8, const F* SQR16, const F* QRT, typename T, const T* LOAD, const T* SAVE> struct GenField
{
    typedef BitsInt<I, B> O;
    typedef LFSR<O, MOD> L;

    static inline constexpr I Sqr1(I a) { return SQR->template Map<O>(a); }
    static inline constexpr I Sqr2(I a) { return SQR2->template Map<O>(a); }
    static inline constexpr I Sqr4(I a) { return SQR4->template Map<O>(a); }
    static inline constexpr I Sqr8(I a) { return SQR8->template Map<O>(a); }
    static inline constexpr I Sqr16(I a) { return SQR16->template Map<O>(a); }

public:
    typedef I Elem;

    inline constexpr int Bits() const { return B; }

    inline constexpr Elem Mul2(Elem val) const { return L::Call(val); }

    inline Elem Mul(Elem a, Elem b) const { return MUL(a, b); }

    class Multiplier
    {
        Elem m_val;
    public:
        inline constexpr explicit Multiplier(const GenField&, Elem a) : m_val(a) {}
        constexpr Elem operator()(Elem a) const { return MUL(m_val, a); }
    };

    /** Compute the square of a. */
    inline constexpr Elem Sqr(Elem val) const { return SQR->template Map<O>(val); }

    /** Compute x such that x^2 + x = a (undefined result if no solution exists). */
    inline constexpr Elem Qrt(Elem val) const { return QRT->template Map<O>(val); }

    /** Compute the inverse of x1. */
    inline Elem Inv(Elem val) const { return InvLadder<I, O, B, MUL, Sqr1, Sqr2, Sqr4, Sqr8, Sqr16>(val); }

    /** Generate a random field element. */
    Elem FromSeed(uint64_t seed) const {
        uint64_t k0 = 0x434c4d554c466c64ull; // "CLMULFld"
        uint64_t k1 = seed;
        uint64_t count = ((uint64_t)B) << 32;
        I ret;
        do {
            ret = O::Mask(I(SipHash(k0, k1, count++)));
        } while(ret == 0);
        return LOAD->template Map<O>(ret);
    }

    Elem Deserialize(BitReader& in) const { return LOAD->template Map<O>(in.Read<B, I>()); }

    void Serialize(BitWriter& out, Elem val) const { out.Write<B, I>(SAVE->template Map<O>(val)); }

    constexpr Elem FromUint64(uint64_t x) const { return LOAD->template Map<O>(O::Mask(I(x))); }
    constexpr uint64_t ToUint64(Elem val) const { return uint64_t(SAVE->template Map<O>(val)); }
};

template<typename I, int B, I MOD, typename F, const F* SQR, const F* SQR2, const F* SQR4, const F* SQR8, const F* SQR16, const F* QRT, typename T, const T* LOAD, const T* SAVE>
using Field = GenField<I, B, MOD, MulWithClMulReduce<I, B, MOD>, F, SQR, SQR2, SQR4, SQR8, SQR16, QRT, T, LOAD, SAVE>;

template<typename I, int B, int POS, typename F, const F* SQR, const F* SQR2, const F* SQR4, const F* SQR8, const F* SQR16, const F* QRT, typename T, const T* LOAD, const T* SAVE>
using FieldTri = GenField<I, B, I(1) + (I(1) << POS), MulTrinomial<I, B, POS>, F, SQR, SQR2, SQR4, SQR8, SQR16, QRT, T, LOAD, SAVE>;

}

#endif
