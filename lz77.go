package webp

import "math/bits"

const (
	maxCopyLen  = 4096
	maxCopyDist = 1<<20 - 120
	hashBits    = 18
)

const (
	refLiteral = iota
	refCache
	refCopy
)

// ref is one symbol of an entropy-coded image: a literal pixel, a color cache
// hit or a backward reference.
type ref struct {
	kind uint8
	val  uint32 // pixel, cache index or copy length
	dist uint32 // copy distance in pixels, later its plane code
}

func hashPair(a, b uint32) uint32 {
	return uint32((uint64(a)<<32 | uint64(b)) * 0x9e3779b97f4a7c15 >> (64 - hashBits))
}

func matchLen(pix []uint32, i, j, limit int) int {
	n := 0
	for n < limit && pix[i+n] == pix[j+n] {
		n++
	}
	return n
}

type matcher struct {
	pix   []uint32
	w     int
	iters int
	head  []int32
	prev  []int32
}

func newMatcher(pix []uint32, w, iters int) *matcher {
	m := &matcher{pix: pix, w: w, iters: iters, head: make([]int32, 1<<hashBits), prev: make([]int32, len(pix))}
	for i := range m.head {
		m.head[i] = -1
	}
	return m
}

func (m *matcher) insert(i int) {
	if i+1 < len(m.pix) {
		h := hashPair(m.pix[i], m.pix[i+1])
		m.prev[i] = m.head[h]
		m.head[h] = int32(i)
	}
}

// find returns the longest earlier match for position i.
func (m *matcher) find(i int) (length, dist int) {
	limit := min(maxCopyLen, len(m.pix)-i)
	if limit < 2 {
		return 0, 0
	}
	try := func(j int) {
		if n := matchLen(m.pix, i, j, limit); n > length {
			length, dist = n, i-j
		}
	}
	if i >= 1 {
		try(i - 1)
	}
	if m.w > 1 && i >= m.w {
		try(i - m.w)
	}
	j := m.head[hashPair(m.pix[i], m.pix[i+1])]
	for n := m.iters; j >= 0 && n > 0 && length < limit; n-- {
		if i-int(j) > maxCopyDist {
			break
		}
		if m.pix[int(j)+length] == m.pix[i+length] {
			try(int(j))
		}
		j = m.prev[j]
	}
	return length, dist
}

const minCopyLen = 3

// backwardRefs splits pix into literals and copies with greedy matching and
// one step of lazy evaluation.
func backwardRefs(pix []uint32, w, iters int) []ref {
	refs := make([]ref, 0, len(pix)/2)
	m := newMatcher(pix, w, iters)

	length, dist := m.find(0)
	for i := 0; i < len(pix); {
		if length >= minCopyLen {
			m.insert(i)
			nl, nd := m.find(i + 1)
			if nl > length+1 {
				refs = append(refs, ref{kind: refLiteral, val: pix[i]})
				i++
				length, dist = nl, nd
				continue
			}
			refs = append(refs, ref{kind: refCopy, val: uint32(length), dist: uint32(dist)})
			for k := i + 1; k < i+length; k++ {
				m.insert(k)
			}
			i += length
		} else {
			refs = append(refs, ref{kind: refLiteral, val: pix[i]})
			m.insert(i)
			i++
		}
		if i < len(pix) {
			length, dist = m.find(i)
		}
	}
	return refs
}

const cacheMultiplier = 0x1e35a7bd

// cacheCosts estimates the entropy-coded size of refs for each color cache
// size from 0 to maxCacheBits bits.
func cacheCosts(pix []uint32, refs []ref, w int) [maxCacheBits + 1]float64 {
	var (
		caches [maxCacheBits + 1][]uint32
		hists  [maxCacheBits + 1]histogram
		costs  [maxCacheBits + 1]float64
	)
	for b := range caches {
		caches[b] = make([]uint32, 1<<b)
		hists[b] = newHistogram(b)
	}

	pos := 0
	for _, r := range refs {
		if r.kind == refCopy {
			code, extra := prefixEncode(r.val)
			dcode, dextra := prefixEncode(planeCode(w, r.dist))
			for b := range hists {
				h := &hists[b]
				h.green[256+code]++
				h.dist[dcode]++
				h.extraBits += float64(extra + dextra)
			}
			for _, p := range pix[pos : pos+int(r.val)] {
				key := p * cacheMultiplier
				for b := 1; b <= maxCacheBits; b++ {
					caches[b][key>>(32-b)] = p
				}
			}
			pos += int(r.val)
			continue
		}

		p := r.val
		key := p * cacheMultiplier
		hists[0].addLiteral(p)
		for b := 1; b <= maxCacheBits; b++ {
			k := key >> (32 - b)
			if caches[b][k] == p {
				hists[b].green[256+24+k]++
			} else {
				hists[b].addLiteral(p)
				caches[b][k] = p
			}
		}
		pos++
	}
	for b := range hists {
		costs[b] = hists[b].cost()
	}
	return costs
}

// applyCache rewrites literals that hit a color cache of the given size.
func applyCache(pix []uint32, refs []ref, cacheBits int) {
	if cacheBits == 0 {
		return
	}
	cache := make([]uint32, 1<<cacheBits)
	shift := 32 - cacheBits
	pos := 0
	for i := range refs {
		r := &refs[i]
		if r.kind == refCopy {
			for _, p := range pix[pos : pos+int(r.val)] {
				cache[p*cacheMultiplier>>shift] = p
			}
			pos += int(r.val)
			continue
		}
		p := r.val
		k := p * cacheMultiplier >> shift
		if cache[k] == p {
			r.kind, r.val = refCache, k
		} else {
			cache[k] = p
		}
		pos++
	}
}

// prefixEncode splits a length or distance into a prefix symbol and extra
// bits.
func prefixEncode(v uint32) (code, nextra uint32) {
	d := v - 1
	if d < 4 {
		return d, 0
	}
	h := uint32(bits.Len32(d) - 1)
	return 2*h + (d>>(h-1))&1, h - 1
}

var planeCodes = func() [128]uint8 {
	var t [128]uint8
	for i, v := range distanceMap {
		t[v] = uint8(i + 1)
	}
	return t
}()

// distanceMap lists the (dy<<4 | 8-dx) offsets of the 120 short distance
// codes, in code order.
var distanceMap = [120]uint8{
	0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
	0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
	0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
	0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
	0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
	0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
	0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
	0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
	0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
	0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
	0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
	0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
}

// planeCode maps a distance in pixels to a VP8L distance code, using the
// short two-dimensional codes for nearby pixels.
func planeCode(w int, dist uint32) uint32 {
	d := int(dist)
	dy, dx := d/w, d%w
	switch {
	case dx <= 8 && dy < 8:
	case dx > w-8 && dy < 7:
		dx -= w
		dy++
	default:
		return dist + 120
	}
	if c := planeCodes[dy<<4|(8-dx)]; c != 0 {
		return uint32(c)
	}
	return dist + 120
}
