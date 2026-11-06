package webp

import (
	"math/bits"
	"sort"
)

// huffmanCode is a canonical prefix code. Codes are stored bit-reversed so
// they can be written LSB first. A code with a single used symbol has a
// header length of 1 but costs zero bits per symbol.
type huffmanCode struct {
	lengths []uint8
	codes   []uint16
	nbits   []uint8
}

func newHuffmanCode(hist []uint32, maxLen int) huffmanCode {
	c := huffmanCode{
		lengths: make([]uint8, len(hist)),
		codes:   make([]uint16, len(hist)),
		nbits:   make([]uint8, len(hist)),
	}
	used := 0
	last := 0
	for s, n := range hist {
		if n > 0 {
			used++
			last = s
		}
	}
	switch used {
	case 0:
		return c
	case 1:
		c.lengths[last] = 1
		return c
	}
	buildLengths(hist, maxLen, c.lengths)
	copy(c.nbits, c.lengths)

	var count [16]uint16
	for _, l := range c.lengths {
		count[l]++
	}
	count[0] = 0
	var next [16]uint16
	code := uint16(0)
	for l := 1; l < 16; l++ {
		code = (code + count[l-1]) << 1
		next[l] = code
	}
	for s, l := range c.lengths {
		if l > 0 {
			c.codes[s] = bits.Reverse16(next[l]) >> (16 - l)
			next[l]++
		}
	}
	return c
}

func (c *huffmanCode) put(w *bitWriter, s int) {
	w.write(uint32(c.codes[s]), uint(c.nbits[s]))
}

type huffNode struct {
	count       uint64
	left, right int32
}

// buildLengths computes Huffman code lengths no longer than maxLen. When the
// optimal tree is too deep, small counts are raised until it fits.
func buildLengths(hist []uint32, maxLen int, lengths []uint8) {
	var syms []int
	for s, n := range hist {
		if n > 0 {
			syms = append(syms, s)
		}
	}
	nodes := make([]huffNode, 0, 2*len(syms))
	depth := make([]uint8, 2*len(syms))

	for floor := uint64(1); ; floor *= 2 {
		nodes = nodes[:0]
		for _, s := range syms {
			nodes = append(nodes, huffNode{count: max(uint64(hist[s]), floor), left: -1, right: int32(s)})
		}
		sort.SliceStable(nodes, func(i, j int) bool { return nodes[i].count < nodes[j].count })

		// Two-queue merge: leaves are sorted and internal nodes are created
		// in non-decreasing order of count.
		nleaf := len(nodes)
		li, ii := 0, nleaf
		pick := func() int32 {
			if li < nleaf && (ii >= len(nodes) || nodes[li].count <= nodes[ii].count) {
				li++
				return int32(li - 1)
			}
			ii++
			return int32(ii - 1)
		}
		for len(nodes) < 2*nleaf-1 {
			a, b := pick(), pick()
			nodes = append(nodes, huffNode{count: nodes[a].count + nodes[b].count, left: a, right: b})
		}

		depth[len(nodes)-1] = 0
		deepest := uint8(0)
		for i := len(nodes) - 1; i >= nleaf; i-- {
			d := depth[i] + 1
			depth[nodes[i].left] = d
			depth[nodes[i].right] = d
			deepest = max(deepest, d)
		}
		if int(deepest) <= maxLen {
			for i := 0; i < nleaf; i++ {
				lengths[nodes[i].right] = depth[i]
			}
			return
		}
	}
}

var codeLengthOrder = [19]uint8{17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// writeHuffmanCode writes the header that lets a decoder rebuild c.
func writeHuffmanCode(w *bitWriter, c *huffmanCode) {
	var syms [3]int
	n := 0
	for s, l := range c.lengths {
		if l > 0 {
			if n < len(syms) {
				syms[n] = s
			}
			n++
		}
	}

	if n == 0 || n <= 2 && syms[n-1] < 256 {
		w.write(1, 1)
		if n == 0 {
			w.write(0, 3)
			return
		}
		w.write(uint32(n-1), 1)
		if syms[0] < 2 {
			w.write(0, 1)
			w.write(uint32(syms[0]), 1)
		} else {
			w.write(1, 1)
			w.write(uint32(syms[0]), 8)
		}
		if n == 2 {
			w.write(uint32(syms[1]), 8)
		}
		return
	}

	w.write(0, 1)

	last := len(c.lengths) - 1
	for c.lengths[last] == 0 {
		last--
	}
	tokens := lengthTokens(c.lengths[:last+1])
	trimmed := len(tokens) >= 2
	if !trimmed {
		tokens = lengthTokens(c.lengths)
	}

	var hist [19]uint32
	for _, t := range tokens {
		hist[t.code]++
	}
	cl := newHuffmanCode(hist[:], 7)

	ncodes := 19
	for ncodes > 4 && cl.lengths[codeLengthOrder[ncodes-1]] == 0 {
		ncodes--
	}
	w.write(uint32(ncodes-4), 4)
	for _, s := range codeLengthOrder[:ncodes] {
		w.write(uint32(cl.lengths[s]), 3)
	}

	if trimmed {
		w.write(1, 1)
		m := uint32(len(tokens) - 2)
		k := 0
		for m>>(2+2*k) != 0 {
			k++
		}
		w.write(uint32(k), 3)
		w.write(m, uint(2+2*k))
	} else {
		w.write(0, 1)
	}

	for _, t := range tokens {
		cl.put(w, int(t.code))
		switch t.code {
		case 16:
			w.write(uint32(t.extra), 2)
		case 17:
			w.write(uint32(t.extra), 3)
		case 18:
			w.write(uint32(t.extra), 7)
		}
	}
}

type lengthToken struct {
	code, extra uint8
}

// lengthTokens run-length encodes code lengths with the repeat codes 16, 17
// and 18.
func lengthTokens(lengths []uint8) []lengthToken {
	var out []lengthToken
	prev := uint8(8)
	for i := 0; i < len(lengths); {
		v := lengths[i]
		run := 1
		for i+run < len(lengths) && lengths[i+run] == v {
			run++
		}
		i += run

		if v == 0 {
			for run > 0 {
				switch {
				case run < 3:
					out = append(out, lengthToken{0, 0})
					run--
				case run <= 10:
					out = append(out, lengthToken{17, uint8(run - 3)})
					run = 0
				default:
					n := min(run, 138)
					out = append(out, lengthToken{18, uint8(n - 11)})
					run -= n
				}
			}
			continue
		}

		if v != prev {
			out = append(out, lengthToken{v, 0})
			prev = v
			run--
		}
		for run > 0 {
			if run < 3 {
				out = append(out, lengthToken{v, 0})
				run--
				continue
			}
			n := min(run, 6)
			out = append(out, lengthToken{16, uint8(n - 3)})
			run -= n
		}
	}
	return out
}
