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

#include "movegen.h"

#include "board.h"
#include "pattern.h"
#include "scopedmove.h"

#include <algorithm>
#include <optional>
#include <simde/x86/sse2.h>

namespace {

/// Max distance to find a pos in a line.
constexpr int MaxFindDist = 4;

// -------------------------------------------------
// Move filters

/// Decide whether a candidate move qualifies for any threat tier requested by GenType, based on
/// its Pattern4 (and, for COMB types, its per-direction line patterns).
template <GenType Type>
inline bool basicPatternFilter(const Board &board, Pos pos, Color side)
{
    Pattern4 p4 = board.pattern4(pos, side);

    if constexpr (bool(Type & WINNING)) {
        if (p4 >= B_FLEX4)
            return true;
    }

    if constexpr (bool(Type & VCF)) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= D_BLOCK4_PLUS)
                return true;
        }
        // RULE_* occupy the low two bits as a selector, so test for renju by equality, not by a
        // bitwise-AND flag check (which would also fire for RULE_FREESTYLE / RULE_STANDARD).
        else if constexpr ((Type & RULE_RENJU) == RULE_RENJU) {
            if (p4 >= E_BLOCK4
                || p4 == FORBID
                       && (board.pattern(pos, side, 0) >= B4 || board.pattern(pos, side, 1) >= B4
                           || board.pattern(pos, side, 2) >= B4
                           || board.pattern(pos, side, 3) >= B4))
                return true;
        }
        else {
            if (p4 >= E_BLOCK4)
                return true;
        }
    }

    if ((Type & VCT) && p4 < E_BLOCK4) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= G_FLEX3_PLUS)
                return true;
        }
        else {
            if (p4 >= H_FLEX3)
                return true;
        }
    }

    if ((Type & VC2) && p4 < H_FLEX3) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= J_FLEX2_2X)
                return true;
        }
        else {
            if (p4 >= L_FLEX2)
                return true;
        }
    }

    return bool(Type & TRIVIAL);
}

/// A fast check function to skip unnecessary move generation process.
/// @return Whether complete move generation should continue.
template <GenType Type>
constexpr bool preCheckFilter(const Board &board, Color side)
{
    if (bool(Type & VCF)) {
        if constexpr (bool(Type & COMB)) {
            if (board.p4Count(side, D_BLOCK4_PLUS) + board.p4Count(side, C_BLOCK4_FLEX3) == 0)
                return false;
        }
        else {
            if (board.p4Count(side, E_BLOCK4) + board.p4Count(side, D_BLOCK4_PLUS)
                    + board.p4Count(side, C_BLOCK4_FLEX3)
                == 0)
                return false;
        }
    }

    return true;
}

// -------------------------------------------------
// Fixed QVCF neighborhood

constexpr int  QvcfRadius   = 4;
constexpr auto QvcfRowMasks = [] {
    std::array<unsigned, 2 * QvcfRadius + 1> rows {};
    for (Direction offset : RANGE_SQUARE2_LINE4) {
        Pos p = Pos(QvcfRadius, QvcfRadius) + offset;
        rows[p.y()] |= 1U << p.x();
    }
    return rows;
}();

ScoredMove *generateQvcfRows(const Board &board, ScoredMove *moveList, Pos center, Pattern4 minimum)
{
    // Even the bottom stencil row has a full padding row after it for the 16-byte load.
    static_assert(BOARD_BOUNDARY > QvcfRadius && FULL_BOARD_SIZE >= 16);
    const int  shift     = 4 * board.sideToMove();
    const auto nibble    = simde_mm_set1_epi8(int8_t(0xf << shift));
    const auto minimumP4 = simde_mm_set1_epi8(int8_t(minimum << shift));
    Pos        row       = center + QvcfRadius * UP_LEFT;
    for (unsigned columns : QvcfRowMasks) {
        const auto pairs = board.pattern4Pairs<16>(row);
        const auto p4    = simde_mm_and_si128(simde_mm_loadu_si128(pairs.data()), nibble);
        const auto hits  = simde_mm_cmpeq_epi8(simde_mm_max_epu8(p4, minimumP4), p4);
        uint64_t   bits  = simde_mm_movemask_epi8(hits) & columns;
        while (bits) {
            Pos pos = row + pop_lsb(bits) * RIGHT;
            // Occupied cells retain their placement-time pattern. Use the raw candidate set here.
            if (board.isEmptyCandidate(pos))
                *moveList++ = pos;
        }
        row += DOWN;
    }
    return moveList;
}

// -------------------------------------------------
// Pattern4 position lookup

/// Get the first found pos that has the given pattern4.
/// @return Pos::NONE if we cannot find the pattern4 pos.
/// @note Board state must satisfy `board.p4Count(side, p4) > 0`.
Pos findFirstPattern4Pos(const Board &board, Color side, Pattern4 p4)
{
    FOR_EVERY_CAND_POS(&board, pos)
    {
        if (board.pattern4(pos, side) == p4)
            return pos;
    }

    assert(false && "can not find pattern4");
    return Pos::NONE;
}

/// Locate the most recent empty cell with `p4` for `side`, trusting the StateInfo lastPattern4
/// marker on the fast path and falling back to a full candidate scan only if the marker is stale.
/// @note p4 must be one of [C_BLOCK4_FLEX3, B_FLEX4, A_FIVE] (the values lastPattern4 tracks).
Pos findPattern4Pos(const Board &board, Color side, Pattern4 p4)
{
    Pos pos = board.stateInfo().lastPattern4(side, p4);
    if (board.get(pos) == EMPTY && board.pattern4(pos, side) == p4)
        return pos;
    return findFirstPattern4Pos(board, side, p4);
}

/// Find all pseudo defence pos of FOUR pattern4.
/// @param side Color of side with a FOUR pattern4.
ScoredMove *findAllPseudoFourDefendPos(const Board &board, Color side, ScoredMove *moveList)
{
    FOR_EVERY_CAND_POS(&board, pos)
    {
        Pattern4 p4 = board.pattern4(pos, side);

        if (p4 >= E_BLOCK4)
            *moveList++ = pos;
        else if (p4 == FORBID) {
            assert(side == BLACK);

            for (int dir = 0; dir < 4; dir++) {
                // Check if this pos is a B4 + (B4/F4), but recognized as FORBID in Renju.
                // If true, this is still a defend four pos for white.
                if (board.pattern(pos, BLACK, dir) >= B4) {
                    *moveList++ = pos;
                    break;
                }
            }
        }
    }

    return moveList;
}

// -------------------------------------------------
// Exact defence reconstruction: opponent flex four (B_FLEX4)

/// Collect the exact defences of a completed flex-four line: the square `f4Pos` that
/// completes the five (pattern F4 in `dir`), plus the adjacent empty extension square(s)
/// of the line, which came from an F3 attack line pattern (_*OOO*_, X*OOO**X, _*O*OO*_,
/// _O*O*O*O_). Appends 2 or 3 defences to `list`.
ScoredMove *findFlex4LineDefence(const Board &board,
                                 Color        oppo,
                                 Pos          f4Pos,
                                 int          dir,
                                 ScoredMove *const list)
{
    assert(board.pattern(f4Pos, oppo, dir) == F4);

    list[0] = f4Pos;  // Add first defence

    Pos pos = f4Pos;
    for (int i = 0; i < MaxFindDist; i++) {
        pos -= DIRECTION[dir];

        if (Color piece = board.get(pos); piece == oppo)
            continue;
        else if (piece == EMPTY) {
            list[1] = pos;  // Second defence
            if (board.pattern(pos, oppo, dir) == F4
                && (board.pattern4(pos, oppo) != FORBID || !board.checkForbiddenPoint(pos)))
                return list + 2;
        }
        break;
    }
    pos = f4Pos;
    for (int i = 0; i < MaxFindDist; i++) {
        pos += DIRECTION[dir];

        if (Color piece = board.get(pos); piece == oppo)
            continue;
        else if (piece == EMPTY) {
            if (board.pattern(pos, oppo, dir) == F4
                && (board.pattern4(pos, oppo) != FORBID || !board.checkForbiddenPoint(pos))) {
                list[1] = pos;  // Second defence
                return list + 2;
            }
            else
                list[2] = pos;  // Third defence
        }
        break;
    }

    return list + 3;
}

/// Collect the defences of a flex four formed by a double B3 attack line pattern
/// (XOOO**_ + XOOO**_): the B4-pattern square `b4Pos` itself plus every nearby empty
/// square that also completes a four, in all four directions.
ScoredMove *findDoubleB3Defence(const Board &board,
                                Color        oppo,
                                Pos          b4Pos,
                                [[maybe_unused]] int dir,
                                ScoredMove  *list)
{
    assert(board.pattern(b4Pos, oppo, dir) == B4 || board.pattern(b4Pos, oppo, dir) == B4S);

    *list++ = b4Pos;

    for (int d = 0; d < 4; d++) {
        int i, j;
        Pos pos = b4Pos;
        for (i = 0; i < MaxFindDist; i++) {
            pos -= DIRECTION[d];

            if (Color piece = board.get(pos); piece == oppo)
                continue;
            else if (piece == EMPTY && board.pattern(pos, oppo, d) >= B4)
                *list++ = pos;
            break;
        }
        pos = b4Pos;
        for (j = MaxFindDist - i; j > 0; j--) {
            pos += DIRECTION[d];

            if (Color piece = board.get(pos); piece == oppo)
                continue;
            else if (piece == EMPTY && board.pattern(pos, oppo, d) >= B4)
                *list++ = pos;
            break;
        }
    }

    return list;
}

/// Scan both ways along `dir` from the F3 attack move for the empty square whose
/// placement completed the flex four (line pattern F4 and aggregate B_FLEX4).
/// @return The completion square, or Pos::NONE if this direction has none.
Pos findFlex4CompletionSquare(const Board &board, Color oppo, Pos attackPos, int dir)
{
    for (Direction step : {-1 * DIRECTION[dir], DIRECTION[dir]}) {
        Pos pos = attackPos;
        for (int i = 0; i < MaxFindDist; i++) {
            pos += step;

            if (Color piece = board.get(pos); piece == oppo)
                continue;
            else if (piece == EMPTY) {
                if (board.pattern(pos, oppo, dir) == F4 && board.pattern4(pos, oppo) == B_FLEX4)
                    return pos;
                continue;
            }
            break;
        }
    }
    return Pos::NONE;
}

/// Find all exact defence pos of opponent FOUR pattern4.
template <bool IncludeLosingMoves>
ScoredMove *findFourDefence(const Board &board, ScoredMove *const moveList)
{
    Color       oppo = ~board.sideToMove();
    ScoredMove *last = moveList;

    assert(board.p4Count(oppo, A_FIVE) == 0);
    assert(board.p4Count(oppo, B_FLEX4) > 0);

    // Try to find the last opponent attack move that caused flex4 pattern,
    // then its pos can be used to find the resulted flex4 moves.
    if (Pos lastFlex4AttackPos = board.stateInfo().lastFlex4AttackMove[oppo]) {
        // If a pattern in any direction is F3(F3S), then last four is cause
        // by at least one F3. Find all F3 defend moves in every directions.
        for (int dir = 0; dir < 4; dir++) {
            // A cell's pattern is not updated after a stone is placed there (frozen at its
            // placement-time value), so we can look up the pattern before placement.
            Pattern lastMovePattern = board.pattern(lastFlex4AttackPos, oppo, dir);
            if (lastMovePattern != F3 && lastMovePattern != F3S)
                continue;

            Pos f4Pos = findFlex4CompletionSquare(board, oppo, lastFlex4AttackPos, dir);
            if (f4Pos == Pos::NONE)
                continue;

            // If there has already a F3 line, the second F3 line means a
            // double F3 pattern which can not be defended.
            if (!IncludeLosingMoves && last > moveList)
                return moveList;

            last = findFlex4LineDefence(board, oppo, f4Pos, dir, last);
        }

        if (last > moveList)
            return last;

        // If patterns in all directions are not F3, then the B_FLEX4 must
        // be formed by double B3 (in two direction or one direction).
        // (These two scans deliberately share one distance budget: the forward
        // scan runs for the MaxFindDist - i steps the backward scan left over.)
        for (int dir = 0; dir < 4; dir++) {
            Pattern attackPattern = board.pattern(lastFlex4AttackPos, oppo, dir);
            if (attackPattern != B3 && attackPattern != B3S)
                continue;

            int i, j, empty;
            Pos pos = lastFlex4AttackPos;
            for (i = 0, empty = 0; i < MaxFindDist; i++) {
                pos -= DIRECTION[dir];

                if (Color piece = board.get(pos); piece == oppo)
                    continue;
                else if (piece == EMPTY) {
                    if (board.pattern4(pos, oppo) >= B_FLEX4) {
                        Pattern pattern = board.pattern(pos, oppo, dir);
                        if (pattern == F4)
                            return findFlex4LineDefence(board, oppo, pos, dir, last);
                        else if (pattern == B4 || pattern == B4S)
                            return findDoubleB3Defence(board, oppo, pos, dir, last);
                    }
                    if (++empty >= 2)
                        break;
                    continue;
                }
                break;
            }
            pos = lastFlex4AttackPos;
            for (j = MaxFindDist - i; j > 0; j--) {
                pos += DIRECTION[dir];

                if (Color piece = board.get(pos); piece == oppo)
                    continue;
                else if (piece == EMPTY) {
                    if (board.pattern4(pos, oppo) >= B_FLEX4) {
                        Pattern pattern = board.pattern(pos, oppo, dir);
                        if (pattern == F4)
                            return findFlex4LineDefence(board, oppo, pos, dir, last);
                        else if (pattern == B4 || pattern == B4S)
                            return findDoubleB3Defence(board, oppo, pos, dir, last);
                    }
                    continue;
                }
                break;
            }
        }
    }

    // Some pattern which is impossible in normal game may appear in analysis mode
    return findAllPseudoFourDefendPos(board, oppo, last);
}

// -------------------------------------------------
// Exact defence reconstruction: opponent block four + flex three (C_BLOCK4_FLEX3)

/// Collect all valid defence moves of the F3 line through `f3Pos` in `dir`, from the
/// defence LUT keyed by the line around it. For Renju black F3 lines, forbidden-point
/// interactions can open one extra defence square on the opposite side of a forbidden
/// defence, which is probed with a temporary black stone at `f3Pos`.
template <Rule R>
ScoredMove *findF3LineDefence(const Board &board,
                              Color        oppo,
                              Pos          f3Pos,
                              int          dir,
                              ScoredMove  *list)
{
    assert(board.pattern(f3Pos, oppo, dir) == F3 || board.pattern(f3Pos, oppo, dir) == F3S);

    uint64_t key         = board.getKeyAt<R>(f3Pos, dir);
    uint32_t defenceMask = PatternConfig::lookupDefenceTable<R>(key, oppo);

    const bool checkRenjuDefence   = R == RENJU && oppo == BLACK;
    Pos        pos                 = f3Pos;
    Pos        leftRenjuDefence    = Pos::NONE;
    Pos        rightRenjuDefence   = Pos::NONE;
    bool       prevFound           = false;
    bool       foundLeftForbidden  = false;
    bool       foundRightForbidden = false;

    // In order to check renju defence, we need to put a black move at f3Pos, so that
    // checkForbiddenPoint works correctly. The guards remove it when this function returns.
    std::optional<ScopedSwitchSide>                                  sideGuard;
    std::optional<ScopedMove<Rule::RENJU, Board::MoveType::NO_EVAL>> moveGuard;
    if (checkRenjuDefence) {
        sideGuard.emplace(board, oppo);
        moveGuard.emplace(board, f3Pos);
    }

    for (int i = 0; i < 4; i++) {
        pos -= DIRECTION[dir];

        if ((defenceMask >> (3 - i)) & 0x1) {
            assert(board.isEmpty(pos));
            *list++   = pos;
            prevFound = true;

            if (checkRenjuDefence)
                foundLeftForbidden = foundLeftForbidden || board.checkForbiddenPoint(pos);
        }
        else if (checkRenjuDefence && prevFound && board.isEmpty(pos)) {
            leftRenjuDefence = pos;
            prevFound        = false;
        }
    }
    pos       = f3Pos;
    prevFound = false;
    for (int i = 0; i < 4; i++) {
        pos += DIRECTION[dir];
        if ((defenceMask >> (4 + i)) & 0x1) {
            assert(board.isEmpty(pos));
            *list++   = pos;
            prevFound = true;

            if (checkRenjuDefence)
                foundRightForbidden = foundRightForbidden || board.checkForbiddenPoint(pos);
        }
        else if (checkRenjuDefence && prevFound && board.isEmpty(pos)) {
            rightRenjuDefence = pos;
            prevFound         = false;
        }
    }

    if (checkRenjuDefence) {
        // If we have found a forbidden move for opponent black, we might have some
        // other possible defences. Normally the defence move should be at the other
        // side of the forbidden point, so we add defence move only in this case.
        if (foundLeftForbidden && rightRenjuDefence)
            *list++ = rightRenjuDefence;
        if (foundRightForbidden && leftRenjuDefence)
            *list++ = leftRenjuDefence;
    }

    return list;
}

/// Locate the empty square that completes the B4 line through `b4Pos` in `dir` into a
/// five. Overline-aware under Standard/Renju, and accepts Renju black squares marked
/// FORBID whose Standard-rules line pattern is still a genuine four.
template <Rule R>
Pos findB4InLine(const Board &board, Color oppo, Pos b4Pos, int dir)
{
    // F4 in Renju is judged as OL, but it should be a valid defence
    auto checkRenjuF4 = [dir, &board](Pos pos) {
        auto      lineKey = board.getKeyAt<Rule::STANDARD>(pos, dir);
        Pattern2x pattern = PatternConfig::lookupPattern<Rule::STANDARD>(lineKey);
        return pattern.patBlack >= B4;
    };
    // Detect overline B4 (which is not a valid B4 point) in Standard/Renju
    auto checkNotOverlineB4 = [b4Pos, oppo, dir, &board](Pos pos) {
        ScopedSwitchSide                        side(board, oppo);
        ScopedMove<R, Board::MoveType::NO_EVAL> probe(board, pos);
        return board.pattern(b4Pos, oppo, dir) == F5;
    };

    // A block four has a single five-completing square. Scan backward first: if the nearest
    // candidate there is an overline (a fake B4 that makes six, not five, under Standard/Renju)
    // the checkNotOverlineB4 gate rejects it and we fall through to the forward scan. Because
    // the completion is unique, reaching the forward scan means the real completion lies that
    // way, so its candidate is always a genuine five and needs no overline gate.
    int i, j;
    Pos pos = b4Pos;
    for (i = 0; i < MaxFindDist; i++) {
        pos -= DIRECTION[dir];

        if (Color piece = board.get(pos); piece == oppo)
            continue;
        else if (piece == EMPTY
                 && (board.pattern(pos, oppo, dir) == B4 || board.pattern(pos, oppo, dir) == B4S
                     || R == RENJU && board.pattern4(pos, oppo) == FORBID && checkRenjuF4(pos))) {
            if (R == FREESTYLE || checkNotOverlineB4(pos))
                return pos;
        }
        break;
    }
    pos = b4Pos;
    for (j = MaxFindDist - i; j > 0; j--) {
        pos += DIRECTION[dir];

        if (Color piece = board.get(pos); piece == oppo)
            continue;
        else if (piece == EMPTY
                 && (board.pattern(pos, oppo, dir) == B4 || board.pattern(pos, oppo, dir) == B4S
                     || R == RENJU && board.pattern4(pos, oppo) == FORBID && checkRenjuF4(pos)))
            return pos;
        break;
    }

    assert(false && "did not find B4 pattern pos in C_BLOCK4_FLEX3");
    return Pos::NONE;
}

/// Collect our own counter-threat squares around the opponent's B4 completion square
/// `b4Pos`: empty squares along `dir` where we hold at least a B3 (or where Renju black's
/// pseudo-forbidden B4 makes any adjacent empty square worth listing).
template <Rule R>
ScoredMove *findAllB3CounterDefence(const Board &board,
                                    Color        oppo,
                                    Pos          b4Pos,
                                    int          dir,
                                    ScoredMove  *list)
{
    Color      self = ~oppo;
    const bool isPseudoForbiddenB4 =
        R == RENJU && self == BLACK && board.pattern4(b4Pos, self) == FORBID;

    Pos pos = b4Pos;
    for (int i = 0; i < MaxFindDist; i++) {
        pos -= DIRECTION[dir];

        if (Color piece = board.get(pos); piece == self)
            continue;
        else if (piece == EMPTY && (isPseudoForbiddenB4 || board.pattern(pos, self, dir) >= B3)) {
            *list++ = pos;
            continue;
        }
        break;
    }
    pos = b4Pos;
    for (int i = 0; i < MaxFindDist; i++) {
        pos += DIRECTION[dir];

        if (Color piece = board.get(pos); piece == self)
            continue;
        else if (piece == EMPTY && (isPseudoForbiddenB4 || board.pattern(pos, self, dir) >= B3)) {
            *list++ = pos;
            continue;
        }
        break;
    }

    return list;
}

/// Find all exact defence pos of opponent B4F3 pattern4.
/// @note If no direct defence is needed, empty move list is returned.
template <Rule R>
ScoredMove *findB4F3Defence(const Board &board, ScoredMove *const moveList)
{
    Color oppo = ~board.sideToMove();

    assert(board.p4Count(oppo, A_FIVE) == 0 && board.p4Count(oppo, B_FLEX4) == 0);
    assert(board.p4Count(oppo, C_BLOCK4_FLEX3) > 0);

    // Get opponent B4F3 pos from the last memorized C-type move (rescans if the marker is stale).
    Pos B4F3Pos = findPattern4Pos(board, oppo, C_BLOCK4_FLEX3);
    assert(board.get(B4F3Pos) == EMPTY);
    assert(board.pattern4(B4F3Pos, oppo) == C_BLOCK4_FLEX3);

    ScoredMove *last = moveList;
    *last++          = B4F3Pos;

    // Iterate all directions to find F3 pattern line and B4 pattern line.
    for (int dir = 0; dir < 4; dir++) {
        Pattern pattern = board.pattern(B4F3Pos, oppo, dir);
        if (pattern == F3 || pattern == F3S)
            last = findF3LineDefence<R>(board, oppo, B4F3Pos, dir, last);
        else if (pattern == B4 || pattern == B4S) {
            Pos b4Pos = findB4InLine<R>(board, oppo, B4F3Pos, dir);

            // If we have a B4 counter defence move, then direct defence
            // to the opponent move is unnecessary.
            if (board.pattern4(b4Pos, ~oppo) >= E_BLOCK4)
                return moveList;

            *last++ = b4Pos;

            for (int d = 0; d < 4; d++)
                last = findAllB3CounterDefence<R>(board, oppo, b4Pos, d, last);
        }
    }

    return last;
}

// -------------------------------------------------
// Defence move-list assembly (sort + dedup + filter)

/// Generate defence moves against the opponent's B_FLEX4, deduplicated and sorted. Our own VCF
/// moves are excluded here; they are produced by a separate generator.
/// @note Board state must satisfy `board.p4Count(oppo, B_FLEX4) > 0`.
template <bool IncludeLosingMoves>
ScoredMove *generateFourDefence(const Board &board, ScoredMove *moveList)
{
    assert(board.p4Count(~board.sideToMove(), B_FLEX4));
    ScoredMove *last = findFourDefence<IncludeLosingMoves>(board, moveList);

    std::sort(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos < n.pos; });
    last = std::unique(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos == n.pos; });

    return std::remove_if(moveList, last, [&](ScoredMove move) {
        assert(board.isEmpty(move));
        assert(board.pattern4(move, ~board.sideToMove()) >= E_BLOCK4
               || board.pattern4(move, ~board.sideToMove()) == FORBID);

        // only adds non-vcf moves
        return board.pattern4(move, board.sideToMove()) >= E_BLOCK4;
    });
}

/// Generate defence moves against the opponent's C_BLOCK4_FLEX3, deduplicated and sorted. Our own
/// VCF moves are excluded (produced by a separate generator). Returns an empty list when no direct
/// defence is needed because we have a B4 counter-defence move.
/// @note Board state must satisfy `board.p4Count(oppo, C_BLOCK4_FLEX3) > 0`.
template <Rule R>
ScoredMove *generateB4F3Defence(const Board &board, ScoredMove *moveList)
{
    assert(board.p4Count(~board.sideToMove(), C_BLOCK4_FLEX3));
    ScoredMove *last = findB4F3Defence<R>(board, moveList);

    // If direct defence is not needed, we simply return empty move list.
    if (last == moveList)
        return moveList;

    std::sort(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos < n.pos; });
    last = std::unique(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos == n.pos; });

    return std::remove_if(moveList, last, [&](ScoredMove move) {
        assert(board.isEmpty(move));
        return board.pattern4(move, board.sideToMove()) >= E_BLOCK4;
    });
}

}  // namespace

// -------------------------------------------------
// Public move generators

template <GenType Type>
ScoredMove *generate(const Board &board, ScoredMove *moveList)
{
    [[maybe_unused]] Color self = board.sideToMove();

    FOR_EVERY_CAND_POS(&board, pos)
    {
        // ALL accepts every candidate (basicPatternFilter falls through to TRIVIAL), so skip the
        // per-move filter entirely on this hot path.
        if constexpr (Type == ALL)
            *moveList++ = pos;
        else if (basicPatternFilter<Type>(board, pos, self))
            *moveList++ = pos;
    }

    return moveList;
}

template ScoredMove *generate<VCF>(const Board &, ScoredMove *);
template ScoredMove *generate<VCF | RULE_RENJU>(const Board &, ScoredMove *);
template ScoredMove *generate<ALL>(const Board &, ScoredMove *);

template <GenType Type>
ScoredMove *generateNeighbors(const Board     &board,
                              ScoredMove      *moveList,
                              Pos              center,
                              const Direction *neighbors,
                              size_t           numNeighbors)
{
    Color self = board.sideToMove();

    if (!preCheckFilter<Type>(board, self))
        return moveList;

    if constexpr (Type == VCF || Type == (VCF | COMB)) {
        if (neighbors == RANGE_SQUARE2_LINE4.data() && numNeighbors == RANGE_SQUARE2_LINE4.size()
            && board.isInBoard(center))
            return generateQvcfRows(board,
                                    moveList,
                                    center,
                                    Type == VCF ? E_BLOCK4 : D_BLOCK4_PLUS);
    }

    for (size_t i = 0; i < numNeighbors; i++) {
        Pos pos = center + neighbors[i];

        if (board.isEmptyCandidate(pos) && basicPatternFilter<Type>(board, pos, self))
            *moveList++ = pos;
    }

    return moveList;
}

template ScoredMove *
generateNeighbors<VCF>(const Board &, ScoredMove *, Pos, const Direction *, size_t);
template ScoredMove *
generateNeighbors<VCF | COMB>(const Board &, ScoredMove *, Pos, const Direction *, size_t);

/// Generate direct winning moves for current side to move.
/// @return The first found winning pos.
/// @note Board state must satisfy `p4Count(self, A_FIVE) + p4Count(self, B_FLEX4) > 0`.
template <>
ScoredMove *generate<WINNING>(const Board &board, ScoredMove *moveList)
{
    Color self = board.sideToMove();

    if (board.p4Count(self, A_FIVE)) {
        *moveList++ = findFirstPattern4Pos(board, self, A_FIVE);
    }
    else if (board.p4Count(self, B_FLEX4)) {
        *moveList++ = findFirstPattern4Pos(board, self, B_FLEX4);
    }
    else {
        assert(false && "no winning moves found");
    }

    return moveList;
}

/// Generate the defence move for opponent A_FIVE pattern4.
/// @return The first found FIVE pattern4 pos.
/// @note Board state must satisfy `board.p4Count(oppo, A_FIVE) > 0`.
template <>
ScoredMove *generate<DEFEND_FIVE>(const Board &board, ScoredMove *moveList)
{
    Color oppo = ~board.sideToMove();
    assert(board.p4Count(oppo, A_FIVE) > 0);

    // Get last opponent A_FIVE directly from state info.
    *moveList = board.stateInfo().lastPattern4(oppo, A_FIVE);
    if (LIKELY(board.get(*moveList) == EMPTY && board.pattern4(*moveList, oppo) == A_FIVE))
        return moveList + 1;

    // In case of weird history, we find the A_FIVE pos by iterating all move candidates.
    FOR_EVERY_CAND_POS(&board, pos)
    {
        if (board.pattern4(pos, oppo) == A_FIVE) {
            *moveList = pos;
            return moveList + 1;
        }
    }
    return moveList;
}

/// Generates defence moves for opponent B_FLEX4 pattern4.
/// All VCF moves of us is excluded. VCF moves should be generated
/// by other generator.
/// @note Board state must satisfy `board.p4Count(oppo, B_FLEX4) > 0`.
template <>
ScoredMove *generate<DEFEND_FOUR>(const Board &board, ScoredMove *moveList)
{
    return generateFourDefence<false>(board, moveList);
}

/// Generates defence moves for opponent B_FLEX4 pattern4.
/// This version also generates all losing moves.
template <>
ScoredMove *generate<DEFEND_FOUR | ALL>(const Board &board, ScoredMove *moveList)
{
    return generateFourDefence<true>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_FREESTYLE>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<FREESTYLE>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_STANDARD>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<STANDARD>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_RENJU>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<RENJU>(board, moveList);
}

bool validateOpponentCMove(const Board &board)
{
    // If threat is caused by White, it must be real threat
    if (board.sideToMove() == BLACK)
        return true;

    // We check black C_BLOCK4_FLEX3 move by making the move,
    // then check if there is any B_FLEX4 on board.
    assert(board.p4Count(BLACK, C_BLOCK4_FLEX3) > 0);
    assert(board.p4Count(BLACK, B_FLEX4) == 0);

    Pos lastB4F3Pos = findPattern4Pos(board, BLACK, C_BLOCK4_FLEX3);

    ScopedSwitchSide                                  side(board, BLACK);
    ScopedMove<Rule::RENJU, Board::MoveType::NO_EVAL> probe(board, lastB4F3Pos);
    return board.p4Count(BLACK, B_FLEX4);
}
