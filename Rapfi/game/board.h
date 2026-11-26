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

#pragma once

#include "../core/hash.h"
#include "../core/platform.h"
#include "../core/pos.h"
#include "../core/types.h"
#include "bitboard.h"
#include "candarea.h"
#include "pattern.h"

#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

namespace Search {
class SearchThread;
}
namespace Evaluation {
class Evaluator;
}

/// Iterate `pos` over every on-board (non-wall) cell, in ascending Pos order.
#define FOR_EVERY_POSITION(board, pos) \
    for (Bitboard::Cursor _posCur((board)->onBoard()); Pos pos = _posCur.next();)

/// Iterate `pos` over every empty cell, in ascending Pos order.
#define FOR_EVERY_EMPTY_POS(board, pos) \
    for (Bitboard::Cursor _emptyCur((board)->emptyCells()); Pos pos = _emptyCur.next();)

/// Iterate `pos` over every cell of the candidate-area rectangle `area` (row-major). Walks each
/// row as a contiguous `Pos` run, since incrementing a `Pos` steps one cell along the row.
#define FOR_EVERY_CANDAREA_POS(pos, area)                                                     \
    for (int8_t _y = (area).y0, _y1 = (area).y1, _x0 = (area).x0, _x1 = (area).x1; _y <= _y1; \
         _y++)                                                                                \
        for (Pos pos {_x0, _y}, _rowEnd {_x1, _y}; pos <= _rowEnd; pos++)

/// Iterate `pos` over every empty candidate cell, in ascending Pos order. candidateIterSet()
/// already excludes occupied cells and cells outside the candidate bounding box, so the body needs
/// no per-cell filtering. The set is a snapshot, so a probing move/undo in the body is safe.
#define FOR_EVERY_CAND_POS(board, pos)                                                       \
    for (Bitboard _candSet = (board)->candidateIterSet(), *_candOnce = &_candSet; _candOnce; \
         _candOnce = nullptr)                                                                \
        for (Bitboard::Cursor _candCur(_candSet); Pos pos = _candCur.next();)

/// Iterate `pos` over every empty candidate cell, in ascending Pos order, WITHOUT the
/// candidate-area box clip that FOR_EVERY_CAND_POS applies (i.e. the full candidate set intersected
/// with empty). Like FOR_EVERY_CAND_POS the set is a snapshot, so a probing move/undo in the body
/// is safe.
#define FOR_EVERY_EMPTY_CAND_POS(board, pos)                                        \
    for (Bitboard _ecSet = (board)->emptyCandidates(), *_ecOnce = &_ecSet; _ecOnce; \
         _ecOnce = nullptr)                                                         \
        for (Bitboard::Cursor _ecCur(_ecSet); Pos pos = _ecCur.next();)

/// Per-ply snapshot of the incremental board state. One StateInfo exists per move played, so a
/// move only has to record what changed and undo() can restore the previous ply cheaply. The
/// candidate set lives on Board and is rewound through a journal anchored by `journalStart`.
struct StateInfo
{
    uint32_t journalStart;  ///< First candidate-journal entry owned by this ply.
    CandArea candArea;      ///< Bounding box of candidate cells at this ply.
    Pos      lastMove;      ///< Move that produced this ply (Pos::PASS for a pass).
    /// Move that first created a flex four for this side (set only on a 0 -> nonzero transition).
    Pos lastFlex4AttackMove[SIDE_NB];
    /// Most recent empty cell reaching C_BLOCK4_FLEX3 / B_FLEX4 / A_FIVE, per side.
    Pos lastPattern4Move[SIDE_NB][3];
    /// Count of empty cells holding each Pattern4, per side.
    uint16_t p4Count[SIDE_NB][PATTERN4_NB];
    Value    valueBlack;  ///< Incremental classical evaluation from black's view.

    /// The most recent empty cell that reached the given Pattern4.
    /// @note p4 must be one of [C_BLOCK4_FLEX3, B_FLEX4, A_FIVE].
    Pos lastPattern4(Color side, Pattern4 p4) const
    {
        assert(p4 >= C_BLOCK4_FLEX3 && p4 <= A_FIVE);
        return lastPattern4Move[side][p4 - C_BLOCK4_FLEX3];
    }

    /// Record pos as the most recent cell reaching p4, if p4 is a tactical pattern.
    void setLastPattern4(Color side, Pattern4 p4, Pos pos)
    {
        if (p4 >= C_BLOCK4_FLEX3)
            lastPattern4Move[side][p4 - C_BLOCK4_FLEX3] = pos;
    }
};

/// The central board-position representation. Per-cell patterns live in separate compact arrays,
/// occupancy in bitboards, and directional line state in bitkeys. Per-ply StateInfo and restore
/// records let move() and undo() update the position incrementally rather than rescanning the
/// board. The copy constructor is explicit (and the implicit copy deleted) to avoid accidental
/// expensive copies.
class Board
{
public:
    /// How much state move()/undo() should maintain. Cheaper modes skip work the caller does not
    /// need, and must be paired (the same MT in the matching undo()).
    enum class MoveType {
        NORMAL,        ///< Update cell, pattern, score, classical eval, and external evaluator.
        NO_EVALUATOR,  ///< As NORMAL but skip the external evaluator. Reserved for a future
                       ///< search mode that relies on the classical eval only (no caller yet).
        NO_EVAL,       ///< Update cell, pattern and score only (no eval at all).
        NO_EVAL_MULTI  ///< As NO_EVAL, but do not flip the side to move.
    };

    /// Allocate a board of the given size using the configured default
    /// candidate range (GeneralCfg.defaultCandidateRange). The position is
    /// uninitialized until newGame() is called.
    /// @param boardSize Size of the board, in range [1, MAX_BOARD_SIZE].
    explicit Board(int boardSize);
    /// Allocate a board of the given size and candidate range. The position is
    /// uninitialized until newGame() is called.
    /// @param boardSize Size of the board, in range [1, MAX_BOARD_SIZE].
    Board(int boardSize, CandidateRange candRange);
    /// Clone a board from another and bind a search thread to the clone.
    /// @param other Board object to clone from.
    /// @param thread Search thread to bind (nullptr for no binding).
    explicit Board(const Board &other, Search::SearchThread *thread);
    Board(const Board &) = delete;

    // ------------------------------------------------------------------------
    // board modifier (and dynamic dispatch version)

    /// Initialize the board to an empty board state of rule R.
    /// @tparam R Rule to initialize the board.
    template <Rule R>
    void newGame();

    /// Play a move and incrementally update the board state.
    /// @param pos Pos to put the next stone. Pos::PASS is allowed.
    /// @tparam R Game rule to use.
    /// @tparam MT How much state to maintain (see MoveType).
    /// @note Recursive pass moves are allowed, but the total number of passes must stay below
    ///     cellCount(). As long as consecutive pass moves are not allowed, this holds.
    template <Rule R, MoveType MT = MoveType::NORMAL>
    void move(Pos pos);

    /// Undo move and rollback the board state.
    /// @tparam R Game rule to use.
    /// @tparam MT Type of this move. Must match the move type used in move().
    template <Rule R, MoveType MT = MoveType::NORMAL>
    void undo();

    /// A dynamic dispatch version of newGame().
    void newGame(Rule rule);
    /// A dynamic dispatch version of move().
    void move(Rule rule, Pos pos);
    /// A dynamic dispatch version of undo().
    void undo(Rule rule);

    // ------------------------------------------------------------------------
    // special helper function

    /// Flip the side to move without recording a ply in the state info. Intended only for local
    /// board-checking routines and must be used in pairs. For an actual pass, use move(Pos::PASS).
    void flipSide() { currentSide = ~currentSide; }

    // ------------------------------------------------------------------------
    // pos-specific info queries

    /// Get color of the piece at pos, decoded from the horizontal bitkey.
    /// Codes: 00 wall, 01 white, 10 black, 11 empty; `code ^ ((~code & 1) << 1)` maps them
    /// onto the Color enum exactly (00->WALL, 01->WHITE, 10->BLACK, 11->EMPTY).
    inline Color get(Pos pos) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        int      x    = pos & 31;  // padded x  (Pos is ((y+B)<<5) | (x+B))
        int      y    = pos >> 5;  // padded y
        uint32_t code = uint32_t(bitKey0[y] >> (2 * x)) & 0b11u;
        return Color(code ^ ((~code & 1u) << 1));
    }

    /// Aggregate four-direction pattern of one side at pos. Occupied/wall cells return
    /// their frozen (placement-time / zero) values: move() never rewrites the pattern state
    /// of a cell once a stone lands on it, and undo() restores it LIFO, so a stone's cell
    /// keeps the exact patterns it had as an empty cell at placement time (the "freeze
    /// invariant"; checkForbiddenPoint relies on it).
    inline Pattern4 pattern4(Pos pos, Color side) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        assert(side == BLACK || side == WHITE);
        return Pattern4((pattern4x2[pos] >> (4 * side)) & 0xF);
    }

    /// Read consecutive packed Pattern4 pairs, with black in the low nibble of each byte.
    template <size_t N>
    std::array<uint8_t, N> pattern4Pairs(Pos first) const
    {
        assert(first >= 0 && size_t(first) + N <= FULL_BOARD_CELL_COUNT);
        std::array<uint8_t, N> pairs;
        std::memcpy(pairs.data(), pattern4x2 + first, N);
        return pairs;
    }

    /// Line-level pattern of one side at pos in direction dir.
    inline Pattern pattern(Pos pos, Color side, int dir) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        assert(side == BLACK || side == WHITE);
        assert(dir >= 0 && dir < 4);
        return side == BLACK ? pattern2xs[pos][dir].patBlack : pattern2xs[pos][dir].patWhite;
    }

    /// Combined four-direction pattern code at pos for one side. The pcode bits are
    /// identical in every fused table variant, so this always reads PCODE.
    template <Color C>
    inline PatternCode pcode(Pos pos) const
    {
        static_assert(C == BLACK || C == WHITE);
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        const Pattern2x *p2x = pattern2xs[pos];
        if constexpr (C == BLACK)
            return PatternConfig::PCODE[p2x[0].patBlack][p2x[1].patBlack][p2x[2].patBlack]
                                       [p2x[3].patBlack]
                                           .pcode();
        else
            return PatternConfig::PCODE[p2x[0].patWhite][p2x[1].patWhite][p2x[2].patWhite]
                                       [p2x[3].patWhite]
                                           .pcode();
    }

    /// Both sides' combined four-direction pattern codes at pos (black first). Loads the
    /// 4-byte pattern2xs row once and extracts the four Pattern2x bytes from the register,
    /// so the two PCODE lookups share one row load instead of issuing four byte loads
    /// each. Values are identical to pcode<BLACK>() / pcode<WHITE>().
    inline std::pair<PatternCode, PatternCode> pcodePair(Pos pos) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        uint32_t row;
        std::memcpy(&row, pattern2xs[pos], sizeof(row));

        Pattern2x a, b, c, d;
        uint8_t   ab = uint8_t(row), bb = uint8_t(row >> 8), cb = uint8_t(row >> 16),
                db = uint8_t(row >> 24);
        std::memcpy(&a, &ab, 1);
        std::memcpy(&b, &bb, 1);
        std::memcpy(&c, &cb, 1);
        std::memcpy(&d, &db, 1);

        return {PatternConfig::PCODE[a.patBlack][b.patBlack][c.patBlack][d.patBlack].pcode(),
                PatternConfig::PCODE[a.patWhite][b.patWhite][c.patWhite][d.patWhite].pcode()};
    }

    /// Check if the pos is in the region of current board size.
    bool isInBoard(Pos pos) const { return pos.isInBoard(boardSize, boardSize); }

    /// Check if the pos is on an empty cell. Wall and occupied cells both read false.
    /// @param pos The pos to query, which is assumed to meet 'pos.valid() == true'.
    bool isEmpty(Pos pos) const { return emptyBB.test(pos); }

    /// Check if the pos is legal (on an empty cell or is a pass move). The PASS check comes first
    /// so a pass never reaches isEmpty()/get(), which would index out of the cell grid for
    /// Pos::PASS.
    bool isLegal(Pos pos) const { return pos.valid() && (pos == Pos::PASS || isEmpty(pos)); }

    /// Whether `pos` is currently a move candidate (some played stone's range covers it). This is
    /// the raw in-range test (no box / empty filtering), as used by neighbor move generation.
    bool isCandidate(Pos pos) const { return candidatesBB.test(pos); }

    /// Whether `pos` is an empty candidate cell: in the raw candidate set and currently empty (no
    /// box clip). Off-board positions read as false, so a neighbor probe needs no range check.
    bool isEmptyCandidate(Pos pos) const { return emptyBB.test(pos) && candidatesBB.test(pos); }

    /// Bitboard of all on-board (non-wall) cells; constant for the board's lifetime.
    const Bitboard &onBoard() const { return onBoardBB; }

    /// Bitboard of all currently-empty cells.
    const Bitboard &emptyCells() const { return emptyBB; }

    /// The set of candidate cells to iterate this ply: in-range AND empty AND inside the candidate
    /// bounding box. Returned by value as a snapshot, so iteration is unaffected by a probing
    /// move/undo in the loop body. Used by FOR_EVERY_CAND_POS.
    Bitboard candidateIterSet() const
    {
        Bitboard bb;
        bb.buildCandSet(candidatesBB, emptyBB, stateInfo().candArea);
        return bb;
    }

    /// The candidate cells to iterate ignoring the bounding box: in-range AND empty (`candidates &
    /// emptyCells`). Like candidateIterSet() but without the candArea clip, so it includes every
    /// empty candidate the board tracks. Returned by value as a snapshot. Used by
    /// FOR_EVERY_EMPTY_CAND_POS.
    Bitboard emptyCandidates() const
    {
        Bitboard bb;
        bb.setIntersect(candidatesBB, emptyBB);
        return bb;
    }

    /// Check if a pos on board is forbidden point in Renju rule.
    /// @param pos Pos to check forbidden. Should be an empty cell.
    /// @return Whether this pos is a real forbidden point.
    bool checkForbiddenPoint(Pos pos) const;

    /// Compute the move-ordering score of an empty cell on demand (cold-path helper; hot
    /// callers batch the same lookups inline): own attacking score plus the opponent's
    /// defensive score.
    Score score(Rule rule, Pos pos, Color side) const;

    /// Query the bitkey at the given pos and direction.
    template <Rule R>
    uint64_t getKeyAt(Pos pos, int dir) const;

    // ------------------------------------------------------------------------
    // general board info queries

    int                    size() const { return boardSize; }
    int                    cellCount() const { return boardCellCount; }
    Pos                    centerPos() const { return {boardSize / 2, boardSize / 2}; }
    Pos                    startPos() const { return {0, 0}; }
    Pos                    endPos() const { return {boardSize - 1, boardSize - 1}; }
    Search::SearchThread  *thisThread() const { return thisThread_; }
    Evaluation::Evaluator *evaluator() const { return evaluator_; }

    // ------------------------------------------------------------------------
    // current board state queries

    int   ply() const { return moveCount; };
    int   nonPassMoveCount() const { return moveCount - passMoveCount(); }
    int   passMoveCount() const { return passCount[BLACK] + passCount[WHITE]; }
    int   passMoveCountOfSide(Color side) const { return passCount[side]; }
    int   movesLeft() const { return boardCellCount - nonPassMoveCount(); };
    Color sideToMove() const { return currentSide; }

    /// Fetch the current board hash key.
    HashKey zobristKey() const { return currentZobristKey ^ Hash::zobristSide[currentSide]; }

    /// Compute the board hash key after a move, without actually making the move.
    HashKey zobristKeyAfter(Pos pos) const
    {
        return currentZobristKey ^ Hash::zobristSide[~currentSide]
               ^ (pos != Pos::PASS ? Hash::zobrist[currentSide][pos] : HashKey {});
    }

    /// Get the current pattern4 accumulate counter for one side.
    uint16_t p4Count(Color side, Pattern4 p4) const { return stateInfo().p4Count[side][p4]; }

    // ------------------------------------------------------------------------
    // history board state queries

    /// Get history move pos according to move index.
    /// @param moveIndex Index of the move, in range [0, ply()).
    inline Pos getHistoryMove(int moveIndex) const
    {
        assert(moveIndex >= 0 && moveIndex < moveCount);
        return stateInfos[moveIndex + 1].lastMove;
    }

    /// Get recent history move pos according to reverse index.
    /// @param reverseIndex Reverse index starting from 0 (0 means the last move).
    inline Pos getRecentMove(int reverseIndex) const
    {
        assert(reverseIndex >= 0);
        int index = moveCount - reverseIndex;
        return index <= 0 ? Pos::NONE : stateInfos[index].lastMove;
    }

    /// Get state info of ply according to reverse index.
    /// @param reverseIndex Reverse index starting from 0 (0 means current state).
    inline const StateInfo &stateInfo(int reverseIndex = 0) const
    {
        assert(0 <= reverseIndex && reverseIndex <= moveCount);
        return stateInfos[moveCount - reverseIndex];
    }

    /// Get the last move played, same as getRecentMove(0).
    Pos getLastMove() const { return getRecentMove(0); }

    /// Get the last actual move (excluding null move) of one side.
    /// @param side Color of side to find move.
    /// @return Pos::NONE if there is no such move, otherwise returns the actual move.
    Pos getLastActualMoveOfSide(Color side) const;

    // ------------------------------------------------------------------------
    // miscellaneous

    /// Expand candidate area around the given pos.
    /// @param pos Pos on board to be expanded.
    /// @param fillDist Distance of filled square that will be expanded.
    /// @param lineDist Distance of line that will be expanded.
    void expandCandArea(Pos pos, int fillDist, int lineDist);

    /// Construct a position string from current board state.
    /// Position string example: `h8h7i7g9`.
    std::string positionString() const;

    /// Trace the current board state (used for debugging).
    std::string trace(Rule rule) const;

private:
    /// State needed to restore one changed cell without table lookups.
    struct CellRestore
    {
        Pattern2x pattern2x;     ///< Pre-update line pattern of the changed direction.
        uint8_t   pattern4Pair;  ///< Pre-update packed pattern4 nibbles (both sides).
    };
    /// A move walks at most 2*5 cells in each of four directions.
    static constexpr int MAX_CELL_RESTORES = 40;

    /// Compact restore record for one move. `touchedMask` maps each saved cell back to its
    /// (step*4 + direction) walk slot; cells whose Pattern2x byte did not change are omitted.
    struct MoveRestore
    {
        uint64_t    touchedMask;
        CellRestore cells[MAX_CELL_RESTORES];
    };

    /// Per-direction line patterns of both colors, one byte per direction, cell-major.
    /// AoS so pcode() reads one 4-byte group; the whole array is 4 KB (L1-resident). Sized to
    /// the padded coordinate space (boundary walls included) so neighbour accesses never need
    /// an in-range check.
    Pattern2x pattern2xs[FULL_BOARD_CELL_COUNT][4];
    /// Packed pattern4 nibbles per cell: black in the low nibble, white in the high one.
    uint8_t pattern4x2[FULL_BOARD_CELL_COUNT];

    /// Pack both sides' Pattern4 into one pattern4x2 byte (black low nibble, white high).
    static constexpr uint8_t packP4Pair(Pattern4 black, Pattern4 white)
    {
        return uint8_t(black) | uint8_t(white) << 4;
    }

    // Per-direction bitkeys: 2 bits per cell, indexed so one line maps to a contiguous run of
    // bits. Each array is indexed by the line's fixed coordinate; the bracket shows bit order.
    uint64_t bitKey0[FULL_BOARD_SIZE];          // horizontal   [RIGHT(MSB) - LEFT(LSB)]
    uint64_t bitKey1[FULL_BOARD_SIZE];          // vertical     [DOWN(MSB) - UP(LSB)]
    uint64_t bitKey2[FULL_BOARD_SIZE * 2 - 1];  // main diag    [UP_RIGHT(MSB) - DOWN_LEFT(LSB)]
    uint64_t bitKey3[FULL_BOARD_SIZE * 2 - 1];  // anti diag    [DOWN_RIGHT(MSB) - UP_LEFT(LSB)]

    int     boardSize;           ///< Side length of the board.
    int     boardCellCount;      ///< Number of playable cells (boardSize^2).
    int     moveCount;           ///< Number of moves played (stones + passes).
    int     passCount[SIDE_NB];  ///< Number of passes by each side.
    Color   currentSide;         ///< The side to move.
    HashKey currentZobristKey;   ///< Zobrist key of the position (side-to-move excluded).
    std::unique_ptr<StateInfo[]>   stateInfos;    ///< Per-ply state history, indexed by ply.
    std::unique_ptr<MoveRestore[]> moveRestores;  ///< Per-ply cell restore records.
    const Direction *candidateRange;  ///< Offset table defining a move's candidate neighborhood.
    uint32_t         candidateRangeSize;  ///< Number of offsets in `candidateRange`.
    uint32_t         candAreaExpandDist;  ///< Candidate-area expansion distance per move.
    /// `candidateRange` precomputed as per-row dx-masks, so move() marks a stone's candidate
    /// neighborhood with a few word-wide ORs instead of a per-offset scatter.
    Bitboard::Stencil candStencil;
    Bitboard onBoardBB;  ///< All non-wall cells; built once in newGame (static for the game).
    Bitboard emptyBB;    ///< All empty cells; one bit flipped per move()/undo().

    /// One 0->1 candidate-bit transition batch: XOR `delta` into `words[wordIdx]` to undo.
    struct CandJournalEntry
    {
        uint16_t wordIdx;
        uint64_t delta;
    };
    /// Live entries have disjoint bits per word, so <= 64 entries/word x 16 words bounds
    /// the journal at 1024 entries — no growth logic needed.
    static constexpr uint32_t CAND_JOURNAL_CAPACITY = Bitboard::NumWords * 64;

    Bitboard candidatesBB;  ///< Candidate cells (board-level; rewound via the journal).
    std::unique_ptr<CandJournalEntry[]> candJournal;
    uint32_t                            journalTop;

    Evaluation::Evaluator *evaluator_;   ///< External evaluator (may be null).
    Search::SearchThread  *thisThread_;  ///< Owning search thread (may be null).

    /// Set `bits` in candidatesBB word `w`, journaling its 0->1 transitions for undo.
    void setCandidateBits(int w, uint64_t bits)
    {
        uint64_t newBits = bits & ~candidatesBB.words[w];
        if (newBits) {
            candidatesBB.words[w] |= newBits;
            assert(journalTop < CAND_JOURNAL_CAPACITY);
            candJournal[journalTop++] = {uint16_t(w), newBits};
        }
    }

    void setBitKey(Pos pos, Color c);
    void flipBitKey(Pos pos, Color c);
};

/// OR color `c`'s bit into all four directional bitkeys at `pos` (used to seed empty cells).
inline void Board::setBitKey(Pos pos, Color c)
{
    assert(c == BLACK || c == WHITE);
    int x = pos.x() + BOARD_BOUNDARY;
    int y = pos.y() + BOARD_BOUNDARY;

    const uint64_t mask = 0x1 + c;
    bitKey0[y] |= mask << (2 * x);
    bitKey1[x] |= mask << (2 * y);
    bitKey2[x + y] |= mask << (2 * x);
    bitKey3[FULL_BOARD_SIZE - 1 - x + y] |= mask << (2 * x);
}

/// Toggle color `c`'s bit in all four directional bitkeys at `pos` (placing or removing a stone).
inline void Board::flipBitKey(Pos pos, Color c)
{
    assert(c == BLACK || c == WHITE);
    int x = pos.x() + BOARD_BOUNDARY;
    int y = pos.y() + BOARD_BOUNDARY;

    assert(x >= 0 && x < FULL_BOARD_SIZE);
    assert(y >= 0 && y < FULL_BOARD_SIZE);
    assert(x + y >= 0 && x + y < 2 * FULL_BOARD_SIZE - 1);
    assert(FULL_BOARD_SIZE - 1 - x + y >= 0
           && FULL_BOARD_SIZE - 1 - x + y < 2 * FULL_BOARD_SIZE - 1);

    const uint64_t mask = 0x1 + c;
    bitKey0[y] ^= mask << (2 * x);
    bitKey1[x] ^= mask << (2 * y);
    bitKey2[x + y] ^= mask << (2 * x);
    bitKey3[FULL_BOARD_SIZE - 1 - x + y] ^= mask << (2 * x);
}

/// Extract the line bitkey centered on `pos` in direction `dir`, rotated so the center cell sits
/// at the table-expected position for pattern lookup.
template <Rule R>
inline uint64_t Board::getKeyAt(Pos pos, int dir) const
{
    assert(dir >= 0 && dir < 4);

    constexpr int L = PatternConfig::HalfLineLen<R>;
    int           x = pos.x() + BOARD_BOUNDARY;
    int           y = pos.y() + BOARD_BOUNDARY;

    switch (dir) {
    default:
    case 0: return rotr(bitKey0[y], 2 * (x - L));
    case 1: return rotr(bitKey1[x], 2 * (y - L));
    case 2: return rotr(bitKey2[x + y], 2 * (x - L));
    case 3: return rotr(bitKey3[FULL_BOARD_SIZE - 1 - x + y], 2 * (x - L));
    }
}

inline void Board::newGame(Rule rule)
{
    switch (rule) {
    case FREESTYLE: return newGame<FREESTYLE>();
    case STANDARD: return newGame<STANDARD>();
    case RENJU: return newGame<RENJU>();
    default: assert(false && "invalid rule");
    }
}

inline void Board::move(Rule rule, Pos pos)
{
    switch (rule) {
    case FREESTYLE: return move<FREESTYLE>(pos);
    case STANDARD: return move<STANDARD>(pos);
    case RENJU: return move<RENJU>(pos);
    default: assert(false && "invalid rule");
    }
}

inline void Board::undo(Rule rule)
{
    switch (rule) {
    case FREESTYLE: return undo<FREESTYLE>();
    case STANDARD: return undo<STANDARD>();
    case RENJU: return undo<RENJU>();
    default: assert(false && "invalid rule");
    }
}
