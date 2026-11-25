/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "mix9svqnnue.h"

#include "mixcommon.h"

#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>
#include <cstring>


namespace Evaluation::mix9svq {

using namespace Evaluation;
using namespace Evaluation::mixnet;

constexpr uint32_t ArchHashBase    = 0x84a071fe;
constexpr int      FeatureDim      = 64;
constexpr int      PolicyDim       = 32;
constexpr int      ValueDim        = 64;
constexpr int      FeatDWConvDim   = 32;
constexpr int      PolicyPWConvDim = 16;
constexpr int      NumHeadBucket   = 1;

template <int OutSize, int InSize>
struct StarBlockWeight
{
    FCWeight<OutSize * 2, InSize> value_corner_up1;
    FCWeight<OutSize * 2, InSize> value_corner_up2;
    FCWeight<OutSize, OutSize>    value_corner_down;
};

struct alignas(64) Weight
{
    // 1  mapping layer
    int16_t  codebook[2][65536][FeatureDim];
    uint16_t mapping_index[2][ShapeNum];
    char     __padding_to_64bytes_0[36];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
            policy_pwconv_layer_l2;

        // 4  Value Group MLP (layer 1,2)
        StarBlockWeight<ValueDim, FeatureDim> value_corner;
        StarBlockWeight<ValueDim, FeatureDim> value_edge;
        StarBlockWeight<ValueDim, FeatureDim> value_center;
        StarBlockWeight<ValueDim, ValueDim>   value_quad;

        // 5  Value MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        FCWeight<ValueDim, ValueDim>                  value_l2;
        FCWeight<4, ValueDim>                         value_l3;

        // 6  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(Weight, feature_dwconv_weight) % 64 == 0);
static_assert(offsetof(Weight, buckets) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_corner) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_l1) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_output_weight) % 16 == 0);
static_assert(sizeof(Weight::HeadBucket) % 64 == 0);

struct alignas(64) ValueSumType
{
    static constexpr int NGroup = 3;

    std::array<int32_t, FeatureDim> global;
    std::array<int32_t, FeatureDim> group[NGroup][NGroup];
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(ValueSumType, global) % 64 == 0);
static_assert(offsetof(ValueSumType, group) % 64 == 0);

class Accumulator : private MixAccumulatorState<ValueSumType, FeatureDim, FeatDWConvDim>
{
public:
    explicit Accumulator(int boardSize) : MixAccumulatorState(boardSize) {}

    /// Init accumulator state to empty board.
    void clear(const Weight &w);
    /// Incrementally update the network state for a stone of pieceColor placed at (x, y).
    void move(const Weight &w, Color pieceColor, int x, int y);
    using MixAccumulatorState::undo;

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Weight &w, PolicyBuffer &policyBuffer);

private:
    // Each perspective owns its accumulator. All nine slots are populated together.
    const Weight::HeadBucket *cachedGroupBucket = nullptr;
    alignas(Alignment) int8_t groupInput[ValueSumType::NGroup][ValueSumType::NGroup][FeatureDim];
    alignas(Alignment) int8_t groupOutput[ValueSumType::NGroup][ValueSumType::NGroup][ValueDim];

    int getBucketIndex() { return 0; }
};

}  // namespace Evaluation::mix9svq

namespace {

using namespace Evaluation::mix9svq;

struct Mix9svqWeightLoader : WeightLoader<mix9svq::Weight>
{
    LargePagePtr<Weight> load(std::istream &in, Evaluation::EmptyLoadArgs args)
    {
        auto w = make_unique_large_page<Weight>();

        read_compressed_mapping(in, *w);
        in.read(reinterpret_cast<char *>(&w->feature_dwconv_weight[0][0]),
                sizeof(w->feature_dwconv_weight));
        in.read(reinterpret_cast<char *>(&w->feature_dwconv_bias[0]),
                sizeof(w->feature_dwconv_bias));
        for (int headIdx = 0; headIdx < NumHeadBucket; headIdx++)
            in.read(reinterpret_cast<char *>(&w->buckets[headIdx]), sizeof(w->buckets[headIdx]));

        if (in && in.peek() == std::ios::traits_type::eof()) {
            preprocess(*w);
            return w;
        }
        else
            return nullptr;
    }

    void read_compressed_mapping(std::istream &in, Weight &w)
    {
        // Read the feature codebook (each codebook is its own 10-bit packed stream)
        for (int mappingIdx = 0; mappingIdx < arraySize(w.codebook); mappingIdx++) {
            auto &codebook = w.codebook[mappingIdx];

            PackedSignedIntStream<10> stream;
            for (int i = 0; i < arraySize(codebook); i++)
                for (int j = 0; j < FeatureDim; j++)
                    codebook[i][j] = stream.next(in);
        }

        // Read the codebook index
        in.read(reinterpret_cast<char *>(&w.mapping_index[0][0]), sizeof(w.mapping_index));
    }

    void preprocess(Weight &w)
    {
        for (int bucketIdx = 0; bucketIdx < NumHeadBucket; bucketIdx++) {
            auto &b = w.buckets[bucketIdx];
            simd::preprocessLinear<PolicyDim * 2, FeatureDim>(b.policy_pwconv_layer_l1.weight);
            simd::preprocessDynamicWeightLinear<PolicyPWConvDim,
                                                PolicyDim,
                                                int16_t,
                                                PolicyDim * 2,
                                                0>(b.policy_pwconv_layer_l2.weight,
                                                   b.policy_pwconv_layer_l2.bias);
            simd::preprocessLinear<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>(
                b.policy_pwconv_layer_l2.weight);
            preprocess(b.value_corner);
            preprocess(b.value_edge);
            preprocess(b.value_center);
            preprocess(b.value_quad);
            simd::preprocessLinear<ValueDim, FeatureDim + ValueDim * 4>(b.value_l1.weight);
            simd::preprocessLinear<ValueDim, ValueDim>(b.value_l2.weight);
            simd::preprocessLinear<4, ValueDim>(b.value_l3.weight);
        }
    }

    template <int OutSize, int InSize>
    void preprocess(StarBlockWeight<OutSize, InSize> &b)
    {
        simd::preprocessLinear<OutSize * 2, InSize>(b.value_corner_up1.weight);
        simd::preprocessLinear<OutSize * 2, InSize>(b.value_corner_up2.weight);
        simd::preprocessLinear<OutSize, OutSize>(b.value_corner_down.weight);
    }
};

static Evaluation::WeightRegistry<StandardHeaderLoader<Mix9svqWeightLoader>> WeightReg;

// We can only use alignment 16 here due to a bug in the weight layout:
// policy_output_weight is only aligned to a 16-byte boundary.
using F32LSAlign16 = simd::detail::VecLoadStore<float, 16, IT>;

template <int OutSize, int InSize>
inline void
starBlock(int8_t output[OutSize], int8_t input[InSize], const StarBlockWeight<OutSize, InSize> &w)
{
    alignas(Alignment) int32_t upi32[OutSize * 2];
    alignas(Alignment) int8_t  up1[OutSize * 2], up2[OutSize * 2];
    simd::linear<OutSize * 2, InSize>(upi32,
                                      input,
                                      w.value_corner_up1.weight,
                                      w.value_corner_up1.bias);
    simd::crelu<OutSize * 2, 128>(up1, upi32);

    simd::linear<OutSize * 2, InSize>(upi32,
                                      input,
                                      w.value_corner_up2.weight,
                                      w.value_corner_up2.bias);
    simd::crelu<OutSize * 2, 128, true>(up2, upi32);

    alignas(Alignment) int8_t dotsum[OutSize];
    simd::dot2<OutSize, 128>(dotsum, up1, up2);

    alignas(Alignment) int32_t outputi32[OutSize];
    simd::linear<OutSize, OutSize, true>(outputi32,
                                         dotsum,
                                         w.value_corner_down.weight,
                                         w.value_corner_down.bias);
    simd::crelu<OutSize, 128>(output, outputi32);
}

}  // namespace

namespace Evaluation::mix9svq {

void Accumulator::clear(const Weight &w)
{
    if (currentVersion == -1) {
        // Init mapConv to bias
        for (int i = 0; i < outerBoardSize * outerBoardSize; i++)
            simd::copy<FeatDWConvDim>(mapConv[i].data(), w.feature_dwconv_bias);
        // Init valueSum to zeros
        auto &valueSum = valueSumTable[0];
        simd::zero<FeatureDim>(valueSum.global.data());
        for (int i = 0; i < ValueSumType::NGroup; i++)
            for (int j = 0; j < ValueSumType::NGroup; j++)
                simd::zero<FeatureDim>(valueSum.group[i][j].data());

        typedef Batch<FeatureDim, int16_t>    FeatB;
        typedef Batch<FeatDWConvDim, int16_t> ConvB;
        typedef Batch<FeatureDim, int32_t>    VSumB;

        for (int y = 0, innerIdx = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++, innerIdx++) {
                // Init mapSum from four directions
                simd::zero<FeatureDim>(mapSum[innerIdx].data());
                for (int dir = 0; dir < 4; dir++) {
                    int mappingIdx  = dir / 2;
                    int shapeIdx    = indexTable[innerIdx][dir];
                    int codebookIdx = w.mapping_index[mappingIdx][shapeIdx];
                    simd::add<FeatureDim>(mapSum[innerIdx].data(),
                                          mapSum[innerIdx].data(),
                                          w.codebook[mappingIdx][codebookIdx]);
                }

                // Init mapConv from mapSum
                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto feature = I16LS::load(mapSum[innerIdx].data() + b * FeatB::RegWidth);
                    feature      = I16Op::max(feature, I16Op::setzero());
                    feature      = I16Op::slli<2>(feature);  // mul 4
                    // Apply feature depthwise conv
                    for (int dy = 0; dy <= 2; dy++) {
                        int yi = y + dy;
                        for (int dx = 0; dx <= 2; dx++) {
                            int xi       = x + dx;
                            int outerIdx = xi + yi * outerBoardSize;

                            auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                            auto  convW     = I16LS::load(convWeightBase + b * ConvB::RegWidth);
                            auto  deltaFeat = I16Op::mulhi(convW, feature);
                            auto  convPtr   = mapConv[outerIdx].data() + b * FeatB::RegWidth;
                            auto  convFeat  = I16LS::load(convPtr);
                            convFeat        = I16Op::add(convFeat, deltaFeat);
                            I16LS::store(convPtr, convFeat);
                        }
                    }
                }

                // Add map feature to map value sum
                for (int b = ConvB::NumBatch; b < FeatB::NumBatch; b++) {
                    auto feature  = I16LS::load(mapSum[innerIdx].data() + b * FeatB::RegWidth);
                    feature       = I16Op::max(feature, I16Op::setzero());
                    auto [v0, v1] = Convert<int16_t, int32_t>::convert(feature);

                    auto addToAccumulator =
                        [&, v0_ = v0, v1_ = v1](std::array<int32_t, FeatureDim> &vSum) {
                            auto vSumPtr = vSum.data() + b * 2 * VSumB::RegWidth;
                            auto vSum0   = I32LS::load(vSumPtr);
                            auto vSum1   = I32LS::load(vSumPtr + VSumB::RegWidth);
                            vSum0        = I32Op::add(vSum0, v0_);
                            vSum1        = I32Op::add(vSum1, v1_);
                            I32LS::store(vSumPtr, vSum0);
                            I32LS::store(vSumPtr + VSumB::RegWidth, vSum1);
                        };
                    addToAccumulator(valueSum.global);
                    addToAccumulator(valueSum.group[groupIndex[y]][groupIndex[x]]);
                }
            }
        }

        // Init valueSum by adding all dwconv value features
        for (int y = 0, outerIdx = outerBoardSize + 1; y < boardSize; y++, outerIdx += 2) {
            for (int x = 0; x < boardSize; x++, outerIdx++) {
                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto feature  = I16LS::load(mapConv[outerIdx].data() + b * ConvB::RegWidth);
                    feature       = I16Op::max(feature, I16Op::setzero());  // relu
                    auto [v0, v1] = Convert<int16_t, int32_t>::convert(feature);

                    auto addToAccumulator =
                        [&, v0_ = v0, v1_ = v1](std::array<int32_t, FeatureDim> &vSum) {
                            auto vSumPtr = vSum.data() + b * 2 * VSumB::RegWidth;
                            auto vSum0   = I32LS::load(vSumPtr);
                            auto vSum1   = I32LS::load(vSumPtr + VSumB::RegWidth);
                            vSum0        = I32Op::add(vSum0, v0_);
                            vSum1        = I32Op::add(vSum1, v1_);
                            I32LS::store(vSumPtr, vSum0);
                            I32LS::store(vSumPtr + VSumB::RegWidth, vSum1);
                        };
                    addToAccumulator(valueSum.global);
                    addToAccumulator(valueSum.group[groupIndex[y]][groupIndex[x]]);
                }
            }
        }
    }

    // Reset version and init version table to be zeros
    currentVersion = 0;
}

void Accumulator::move(const Weight &w, Color pieceColor, int x, int y)
{
    assert(pieceColor == BLACK || pieceColor == WHITE);

    // Copy version info to the next ply
    const int       innerBoardSizeSqr       = boardSize * boardSize;
    const int       outerBoardSizeSqr       = outerBoardSize * outerBoardSize;
    const int       innerVersionIdxBasePrev = currentVersion * innerBoardSizeSqr;
    const int       outerVersionIdxBasePrev = currentVersion * outerBoardSizeSqr;
    const int       innerVersionIdxBase     = innerVersionIdxBasePrev + innerBoardSizeSqr;
    const int       outerVersionIdxBase     = outerVersionIdxBasePrev + outerBoardSizeSqr;
    const ChangeNum changeNum               = versionChangeNumTable[currentVersion];
    std::copy_n(versionInnerIndexTable + innerVersionIdxBasePrev,
                innerBoardSizeSqr,
                versionInnerIndexTable + innerVersionIdxBase);
    std::copy_n(versionOuterIndexTable + outerVersionIdxBasePrev,
                outerBoardSizeSqr,
                versionOuterIndexTable + outerVersionIdxBase);

    typedef Batch<FeatureDim, int16_t>    FeatB;
    typedef Batch<FeatDWConvDim, int16_t> ConvB;
    typedef Batch<FeatureDim, int32_t>    VSumB;

    // Subtract value feature sum
    int x0            = std::max(x - 6 + 1, 1);
    int y0            = std::max(y - 6 + 1, 1);
    int x1            = std::min(x + 6 + 1, boardSize);
    int y1            = std::min(y + 6 + 1, boardSize);
    int newMapConvIdx = changeNum.outer;
    for (int yi = y0, outerIdxBase = y0 * outerBoardSize; yi <= y1;
         yi++, outerIdxBase += outerBoardSize) {
        for (int xi = x0; xi <= x1; xi++) {
            int outerIdx                                           = xi + outerIdxBase;
            versionOuterIndexTable[outerVersionIdxBase + outerIdx] = newMapConvIdx;
            for (int b = 0; b < ConvB::NumBatch; b++)
                I16LS::store(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth, I16Op::setzero());
            newMapConvIdx++;
        }
    }

    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
        int16_t  mappingIdx;
        int16_t  oldMapIdx;
        int16_t  newMapIdx;
        uint16_t oldCodebookIdx;
        uint16_t newCodebookIdx;
    } changeTable[4 * 11];
    int changeCount = 0;
    int dPower3     = pieceColor + 1;

    // Update shape table and record changes
    const int boardSizeSub1 = boardSize - 1;
    int       newMapIdx     = changeNum.inner;
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = -5; dist <= 5; dist++) {
            int xi = x + dist * DX[dir];
            int yi = y + dist * DY[dir];

            // branchless test: xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize
            if ((xi | (boardSizeSub1 - xi) | yi | (boardSizeSub1 - yi)) < 0)
                continue;

            int             innerIdx = boardSize * yi + xi;
            OnePointChange &c        = changeTable[changeCount++];
            c.x                      = xi;
            c.y                      = yi;
            c.mappingIdx             = dir / 2;  // 0,1 -> 0; 2,3 -> 1
            c.oldMapIdx              = versionInnerIndexTable[innerVersionIdxBase + innerIdx];
            c.newMapIdx              = newMapIdx;

            uint32_t oldShape     = indexTable[c.oldMapIdx][dir];
            indexTable[newMapIdx] = indexTable[c.oldMapIdx];
            uint32_t newShape = indexTable[newMapIdx][dir] = oldShape + dPower3 * Power3[dist + 5];
            assert(newShape < ShapeNum);

            c.oldCodebookIdx = w.mapping_index[c.mappingIdx][oldShape];
            c.newCodebookIdx = w.mapping_index[c.mappingIdx][newShape];

            versionInnerIndexTable[innerVersionIdxBase + innerIdx] = newMapIdx++;
        }
    }

    // Init value sum accumulator. Only per-group sums are accumulated here; the global sum is the
    // invariant total of the 9 groups and is computed from them once at the end of the move.
    I32Op::R vSumGroup[ValueSumType::NGroup][ValueSumType::NGroup][VSumB::NumBatch];
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++)
                vSumGroup[i][j][b] = I32Op::setzero();

    // Incremental update feature sum
    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];
        if (i + 1 < changeCount) {
            const OnePointChange &cnext = changeTable[i + 1];
            multiPrefetch<sizeof(int16_t) * FeatureDim>(
                w.codebook[cnext.mappingIdx][cnext.oldCodebookIdx]);
            multiPrefetch<sizeof(int16_t) * FeatureDim>(
                w.codebook[cnext.mappingIdx][cnext.newCodebookIdx]);
        }

        // Update mapSum
        I16Op::R oldFeats[FeatB::NumBatch];
        I16Op::R newFeats[FeatB::NumBatch];
        for (int b = 0; b < FeatB::NumBatch; b++) {
            auto oldMapFeat =
                I16LS::load(w.codebook[c.mappingIdx][c.oldCodebookIdx] + b * FeatB::RegWidth);
            auto newMapFeat =
                I16LS::load(w.codebook[c.mappingIdx][c.newCodebookIdx] + b * FeatB::RegWidth);
            oldFeats[b] = I16LS::load(mapSum[c.oldMapIdx].data() + b * FeatB::RegWidth);
            newFeats[b] = I16Op::sub(oldFeats[b], oldMapFeat);
            newFeats[b] = I16Op::add(newFeats[b], newMapFeat);
            I16LS::store(mapSum[c.newMapIdx].data() + b * FeatB::RegWidth, newFeats[b]);
            oldFeats[b] = I16Op::max(oldFeats[b], I16Op::setzero());
            newFeats[b] = I16Op::max(newFeats[b], I16Op::setzero());
        }

        // Update mapConv
        for (int b = 0; b < ConvB::NumBatch; b++) {
            oldFeats[b] = I16Op::slli<2>(oldFeats[b]);  // mul 4
            newFeats[b] = I16Op::slli<2>(newFeats[b]);  // mul 4
        }
        for (int dy = 0, outerIdxBase = c.y * outerBoardSize + c.x; dy <= 2;
             dy++, outerIdxBase += outerBoardSize) {
            for (int dx = 0; dx <= 2; dx++) {
                int   outerIdx       = dx + outerIdxBase;
                int   mapConvIdx     = versionOuterIndexTable[outerVersionIdxBase + outerIdx];
                auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                auto *convBase       = mapConv[mapConvIdx].data();

                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto convW      = I16LS::load(convWeightBase + b * ConvB::RegWidth);
                    auto deltaConvF = I16Op::sub(I16Op::mulhi(convW, newFeats[b]),
                                                 I16Op::mulhi(convW, oldFeats[b]));

                    auto convPtr  = convBase + b * ConvB::RegWidth;
                    auto oldConvF = I16LS::load(convPtr);
                    auto newConvF = I16Op::add(oldConvF, deltaConvF);
                    I16LS::store(convPtr, newConvF);
                }
            }
        }

        // Update valueSum
        for (int b = ConvB::NumBatch; b < FeatB::NumBatch; b++) {
            auto deltaF             = I16Op::sub(newFeats[b], oldFeats[b]);
            auto [deltaF0, deltaF1] = Convert<int16_t, int32_t>::convert(deltaF);

            const int offset   = 2 * b;
            auto     &vGroup   = vSumGroup[groupIndex[c.y]][groupIndex[c.x]];
            vGroup[offset + 0] = I32Op::add(vGroup[offset + 0], deltaF0);
            vGroup[offset + 1] = I32Op::add(vGroup[offset + 1], deltaF1);
        }
    }

    // Add value feature sum
    newMapConvIdx = changeNum.outer;
    for (int yi = y0, outerIdxBase = y0 * outerBoardSize; yi <= y1;
         yi++, outerIdxBase += outerBoardSize) {
        int i = groupIndex[yi - 1];
        for (int xi = x0; xi <= x1; xi++) {
            int j             = groupIndex[xi - 1];
            int outerIdx      = xi + outerIdxBase;
            int oldMapConvIdx = versionOuterIndexTable[outerVersionIdxBasePrev + outerIdx];
            for (int b = 0; b < ConvB::NumBatch; b++) {
                auto oldConvF   = I16LS::load(mapConv[oldMapConvIdx].data() + b * ConvB::RegWidth);
                auto deltaConvF = I16LS::load(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth);
                auto newConvF   = I16Op::add(oldConvF, deltaConvF);
                I16LS::store(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth, newConvF);
                oldConvF      = I16Op::max(oldConvF, I16Op::setzero());  // relu
                newConvF      = I16Op::max(newConvF, I16Op::setzero());  // relu
                auto deltaF   = I16Op::sub(newConvF, oldConvF);
                auto [v0, v1] = Convert<int16_t, int32_t>::convert(deltaF);

                const int offset            = 2 * b;
                vSumGroup[i][j][offset + 0] = I32Op::add(vSumGroup[i][j][offset + 0], v0);
                vSumGroup[i][j][offset + 1] = I32Op::add(vSumGroup[i][j][offset + 1], v1);
            }
            newMapConvIdx++;
        }
    }

    // Move to next version
    currentVersion++;
    versionChangeNumTable[currentVersion] = {uint16_t(newMapIdx), uint16_t(newMapConvIdx)};

    // Store value sum: first the per-group sums, then set the global sum to their total. The global
    // sum is the invariant sum of the 9 groups, so deriving it here once is bit-identical to (and
    // cheaper than) propagating a separate global accumulator per changed cell above.
    auto &valueSumOld = valueSumTable[currentVersion - 1];
    auto &valueSumNew = valueSumTable[currentVersion];
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++) {
                auto vOld = I32LS::load(valueSumOld.group[i][j].data() + b * VSumB::RegWidth);
                auto vNew = I32Op::add(vOld, vSumGroup[i][j][b]);
                I32LS::store(valueSumNew.group[i][j].data() + b * VSumB::RegWidth, vNew);
            }
    for (int b = 0; b < VSumB::NumBatch; b++) {
        auto acc = I32Op::setzero();
        for (int i = 0; i < ValueSumType::NGroup; i++)
            for (int j = 0; j < ValueSumType::NGroup; j++)
                acc = I32Op::add(acc,
                                 I32LS::load(valueSumNew.group[i][j].data() + b * VSumB::RegWidth));
        I32LS::store(valueSumNew.global.data() + b * VSumB::RegWidth, acc);
    }
}

std::tuple<float, float, float> Accumulator::evaluateValue(const Weight &w)
{
    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // convert value sum from int32 to int8
    // global feature sum
    alignas(Alignment) int8_t layer0[FeatureDim + ValueDim * 4];
    simd::crelu<FeatureDim, 256, true>(layer0, valueSum.global.data());

    const bool sameBucket = cachedGroupBucket == &bucket;
    auto evaluateGroup = [&](int y, int x, const StarBlockWeight<ValueDim, FeatureDim> &weights) {
        alignas(Alignment) int8_t input[FeatureDim];
        simd::crelu<FeatureDim, 32, true>(input, valueSum.group[y][x].data());
        auto &previous = groupInput[y][x];
        if (sameBucket && std::memcmp(input, previous, sizeof(input)) == 0)
            return;

        std::memcpy(previous, input, sizeof(input));
        starBlock<ValueDim, FeatureDim>(groupOutput[y][x], input, weights);
    };
    evaluateGroup(0, 0, bucket.value_corner);
    evaluateGroup(0, 2, bucket.value_corner);
    evaluateGroup(2, 0, bucket.value_corner);
    evaluateGroup(2, 2, bucket.value_corner);
    evaluateGroup(0, 1, bucket.value_edge);
    evaluateGroup(1, 0, bucket.value_edge);
    evaluateGroup(1, 2, bucket.value_edge);
    evaluateGroup(2, 1, bucket.value_edge);
    evaluateGroup(1, 1, bucket.value_center);
    cachedGroupBucket = &bucket;

    // average pooling
    alignas(Alignment) int8_t group2[2][2][ValueDim];
    using I8B = Batch<ValueDim, int8_t>;
    for (int b = 0; b < I8B::NumBatch; b++) {
        auto v00 = I8LS::load(groupOutput[0][0] + b * I8B::RegWidth);
        auto v01 = I8LS::load(groupOutput[0][1] + b * I8B::RegWidth);
        auto v02 = I8LS::load(groupOutput[0][2] + b * I8B::RegWidth);
        auto v10 = I8LS::load(groupOutput[1][0] + b * I8B::RegWidth);
        auto v11 = I8LS::load(groupOutput[1][1] + b * I8B::RegWidth);
        auto v12 = I8LS::load(groupOutput[1][2] + b * I8B::RegWidth);
        auto v20 = I8LS::load(groupOutput[2][0] + b * I8B::RegWidth);
        auto v21 = I8LS::load(groupOutput[2][1] + b * I8B::RegWidth);
        auto v22 = I8LS::load(groupOutput[2][2] + b * I8B::RegWidth);

        auto q00 = I8Op::avg(I8Op::avg(v00, v01), I8Op::avg(v10, v11));
        auto q01 = I8Op::avg(I8Op::avg(v01, v02), I8Op::avg(v11, v12));
        auto q10 = I8Op::avg(I8Op::avg(v10, v11), I8Op::avg(v20, v21));
        auto q11 = I8Op::avg(I8Op::avg(v11, v12), I8Op::avg(v21, v22));

        I8LS::store(group2[0][0] + b * I8B::RegWidth, q00);
        I8LS::store(group2[0][1] + b * I8B::RegWidth, q01);
        I8LS::store(group2[1][0] + b * I8B::RegWidth, q10);
        I8LS::store(group2[1][1] + b * I8B::RegWidth, q11);
    }

    // quadrant linear layer
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 0 * ValueDim,
                                  group2[0][0],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 1 * ValueDim,
                                  group2[0][1],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 2 * ValueDim,
                                  group2[1][0],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 3 * ValueDim,
                                  group2[1][1],
                                  bucket.value_quad);

    // linear 1
    alignas(Alignment) int32_t layer1i32[ValueDim];
    alignas(Alignment) int8_t  layer1[ValueDim];
    simd::linear<ValueDim, FeatureDim + ValueDim * 4>(layer1i32,
                                                      layer0,
                                                      bucket.value_l1.weight,
                                                      bucket.value_l1.bias);
    simd::crelu<ValueDim, 128>(layer1, layer1i32);

    // linear 2
    alignas(Alignment) int32_t layer2i32[ValueDim];
    alignas(Alignment) int8_t  layer2[ValueDim];
    simd::linear<ValueDim, ValueDim>(layer2i32,
                                     layer1,
                                     bucket.value_l2.weight,
                                     bucket.value_l2.bias);
    simd::crelu<ValueDim, 128>(layer2, layer2i32);

    // linear 3 final
    alignas(Alignment) int32_t layer3i32[4];
    simd::linear<4, ValueDim>(layer3i32, layer2, bucket.value_l3.weight, bucket.value_l3.bias);

    const float scale = 1.0f / (128 * 128);
    return {layer3i32[0] * scale, layer3i32[1] * scale, layer3i32[2] * scale};
}

void Accumulator::evaluatePolicy(const Weight &w, PolicyBuffer &policyBuffer)
{
    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    alignas(Alignment) int8_t layer0[FeatureDim];
    simd::crelu<FeatureDim, 256, true>(layer0, valueSum.global.data());

    // policy pwconv weight layer
    alignas(Alignment) int32_t layer1i32[PolicyDim * 2];
    alignas(Alignment) int8_t  layer1[PolicyDim * 2];
    simd::linear<PolicyDim * 2, FeatureDim>(layer1i32,
                                            layer0,
                                            bucket.policy_pwconv_layer_l1.weight,
                                            bucket.policy_pwconv_layer_l1.bias);
    simd::crelu<PolicyDim * 2, 128>(layer1, layer1i32);

    alignas(Alignment) int32_t layer2i32[PolicyPWConvDim * PolicyDim + PolicyPWConvDim];
    alignas(Alignment) int16_t pwconvWeighti16[PolicyPWConvDim * PolicyDim];
    alignas(Alignment) int32_t pwconvBiasi32[PolicyPWConvDim];
    simd::linear<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>(
        layer2i32,
        layer1,
        bucket.policy_pwconv_layer_l2.weight,
        bucket.policy_pwconv_layer_l2.bias);
    simd::crelu<PolicyPWConvDim * PolicyDim, 1, true>(pwconvWeighti16, layer2i32);
    // To get pwconv bias, we need to scale the output of layer2 by 128
    {
        typedef Batch<PolicyPWConvDim, int32_t> B;
        for (int i = 0; i < B::NumBatch; i++) {
            auto data = I32LS::load(layer2i32 + PolicyPWConvDim * PolicyDim + i * B::RegWidth);
            data      = I32Op::slli<7>(data);  // scale by 128
            I32LS::store(pwconvBiasi32 + i * B::RegWidth, data);
        }
    }

    const int outerVersionIdxBase = currentVersion * outerBoardSize * outerBoardSize;
    for (int y = 0, innerIdx = 0, outerIdx = outerBoardSize + 1; y < boardSize;
         y++, outerIdx += 2) {
        for (int x = 0; x < boardSize; x++, innerIdx++, outerIdx++) {
            if (!policyBuffer.getComputeFlag(innerIdx))
                continue;

            // Get mapConv index of current version at this point
            int mapConvIdx = versionOuterIndexTable[outerVersionIdxBase + outerIdx];

            // Compute dynamic point-wise policy conv
            static_assert(PolicyDim <= FeatDWConvDim,
                          "Assume PolicyDim <= FeatDWConvDim in evaluatePolicy()!");
            alignas(Alignment) int32_t policyLayer1i32[PolicyPWConvDim];
            simd::linear<PolicyPWConvDim, PolicyDim, false, true, true>(policyLayer1i32,
                                                                        mapConv[mapConvIdx].data(),
                                                                        pwconvWeighti16,
                                                                        pwconvBiasi32);

            // Apply relu, convert to float and accumulate all channels of pwconv feature
            typedef Batch<PolicyPWConvDim, float> PWConvB;
            auto                                  policyAccum = F32Op::setzero();
            for (int i = 0; i < PWConvB::NumBatch; i++) {
                auto featI32 = I32LS::load(policyLayer1i32 + i * PWConvB::RegWidth);
                featI32      = I32Op::max(featI32, I32Op::setzero());
                auto featF32 = Convert<int32_t, float>::convert1(featI32);
                auto outputW =
                    F32LSAlign16::load(bucket.policy_output_weight + i * PWConvB::RegWidth);
                policyAccum  = F32Op::fmadd(featF32, outputW, policyAccum);
            }

            float policy           = F32Op::reduceadd(policyAccum) + bucket.policy_output_bias;
            policyBuffer(innerIdx) = policy;
        }
    }
}

Evaluator::Evaluator(int                   boardSize,
                     Rule                  rule,
                     Numa::NumaNodeId      numaNodeId,
                     std::filesystem::path blackWeightPath,
                     std::filesystem::path whiteWeightPath)
    : Evaluation::Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
{
    constexpr uint32_t ArchHash = ArchHashBase
                                  ^ (((FeatDWConvDim / 8) << 20) | ((ValueDim / 8) << 14)
                                     | ((PolicyDim / 8) << 8) | (FeatureDim / 8));
    loadWeightPair<CompressedWrapper<StandardHeaderLoader<Mix9svqWeightLoader>>>(WeightReg,
                                                                                 weight,
                                                                                 "mix9svq nnue",
                                                                                 ArchHash,
                                                                                 boardSize,
                                                                                 rule,
                                                                                 numaNodeId,
                                                                                 blackWeightPath,
                                                                                 whiteWeightPath);

    accumulator[BLACK] = std::make_unique<Accumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Accumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);
}

Evaluator::~Evaluator()
{
    if (weight[BLACK])
        WeightReg.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        WeightReg.unloadWeight(weight[WHITE]);
}

void Evaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK]);
    accumulator[WHITE]->clear(*weight[WHITE]);
}

void Evaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Evaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Evaluator::evaluateValue(const Board &board, AccLevel level)
{
    Color self = board.sideToMove();

    // Apply all incremental update for both sides and calculate value
    flushMoveCache(self);
    auto [win, loss, draw] = accumulator[self]->evaluateValue(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    flushMoveCache(self);
    accumulator[self]->evaluatePolicy(*weight[self], policyBuffer);
}

void Evaluator::flushMoveCache(Color side)
{
    applyMoveCache(side, moveCache[side], *accumulator[side], *weight[side]);
}

void Evaluator::addCache(Color side, int x, int y, bool isUndo)
{
    queueMoveCache(moveCache, side, x, y, isUndo, boardSize * boardSize);
}

}  // namespace Evaluation::mix9svq
