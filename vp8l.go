package webp

import "math"

const maxCacheBits = 10

// histogram counts the symbols of the five prefix codes. The parts share
// one backing array so whole histograms can be added cheaply.
type histogram struct {
	all       []uint32
	green     []uint32
	red       []uint32
	blue      []uint32
	alpha     []uint32
	dist      []uint32
	extraBits float64
}

func newHistogram(cacheBits int) histogram {
	n := 256 + 24
	if cacheBits > 0 {
		n += 1 << cacheBits
	}
	all := make([]uint32, n+3*256+40)
	return histogram{
		all:   all,
		green: all[:n],
		red:   all[n : n+256],
		blue:  all[n+256 : n+512],
		alpha: all[n+512 : n+768],
		dist:  all[n+768:],
	}
}

func (h *histogram) parts() [5][]uint32 {
	return [5][]uint32{h.green, h.red, h.blue, h.alpha, h.dist}
}

func (h *histogram) addLiteral(p uint32) {
	h.alpha[p>>24]++
	h.red[p>>16&0xff]++
	h.green[p>>8&0xff]++
	h.blue[p&0xff]++
}

// add counts r, whose copy distance must already be a plane code.
func (h *histogram) add(r ref) {
	switch r.kind {
	case refLiteral:
		h.addLiteral(r.val)
	case refCache:
		h.green[256+24+r.val]++
	case refCopy:
		code, extra := prefixEncode(r.val)
		dcode, dextra := prefixEncode(r.dist)
		h.green[256+code]++
		h.dist[dcode]++
		h.extraBits += float64(extra + dextra)
	}
}

func (h *histogram) cost() float64 {
	c := h.extraBits
	for _, p := range h.parts() {
		c += entropy(p)
	}
	return c
}

// entropy returns the Shannon cost in bits of coding the samples in hist.
func entropy(hist []uint32) float64 {
	var total, sum float64
	for _, n := range hist {
		if n > 0 {
			f := float64(n)
			total += f
			sum += f * math.Log2(f)
		}
	}
	if total == 0 {
		return 0
	}
	return total*math.Log2(total) - sum
}

// transformBits is the log2 tile size of the predictor and cross-color
// transforms.
const transformBits = 5

// effort controls the speed and size trade-off of lossless encoding.
type effort struct {
	iters  int  // hash chain candidates per position
	tryAll bool // try every strategy, not just the likely best
}

func newEffort(q float32) effort {
	return effort{iters: 8 + int(q), tryAll: q >= 80}
}

// encodeLossless returns a complete VP8L bitstream.
func encodeLossless(argb []uint32, w, h int, q float32) []byte {
	var hasAlpha uint32
	for _, p := range argb {
		if p>>24 != 0xff {
			hasAlpha = 1
			break
		}
	}
	bw := &bitWriter{}
	bw.write(0x2f, 8)
	bw.write(uint32(w-1), 14)
	bw.write(uint32(h-1), 14)
	bw.write(hasAlpha, 1)
	bw.write(0, 3)
	return append(bw.bytes(), encodeLosslessBody(argb, w, h, newEffort(q))...)
}

// encodeLosslessBody encodes the image with each applicable strategy and
// keeps the smallest result. The body starts on a byte boundary, both after
// the VP8L header and inside an ALPH chunk.
func encodeLosslessBody(argb []uint32, w, h int, e effort) []byte {
	var best []byte
	keep := func(b []byte) {
		if best == nil || len(b) < len(best) {
			best = b
		}
	}

	pal, ok := buildPalette(argb)
	if ok {
		keep(encodePalette(argb, w, h, pal, e))
	}
	if !ok || e.tryAll {
		keep(encodeTransformed(argb, w, h, e))
	}
	if e.tryAll || w*h <= 64*64 {
		keep(encodePlain(argb, w, h, e))
	}
	return best
}

func encodePlain(argb []uint32, w, h int, e effort) []byte {
	bw := &bitWriter{}
	bw.write(0, 1)
	encodeImage(bw, argb, w, h, true, e)
	return bw.bytes()
}

func encodeTransformed(argb []uint32, w, h int, e effort) []byte {
	pix := make([]uint32, len(argb))
	copy(pix, argb)
	bw := &bitWriter{}

	subtractGreen(pix)
	bw.write(1, 1)
	bw.write(transformSubtractGreen, 2)

	bits := transformBits
	tw, th := tiles(w, bits), tiles(h, bits)

	res, modes := predict(pix, w, h, bits)
	bw.write(1, 1)
	bw.write(transformPredictor, 2)
	bw.write(uint32(bits-2), 3)
	encodeImage(bw, modes, tw, th, false, e)

	elems := crossColor(res, w, h, bits)
	bw.write(1, 1)
	bw.write(transformCrossColor, 2)
	bw.write(uint32(bits-2), 3)
	encodeImage(bw, elems, tw, th, false, e)

	bw.write(0, 1)
	encodeImage(bw, res, w, h, true, e)
	return bw.bytes()
}

func encodePalette(argb []uint32, w, h int, pal []uint32, e effort) []byte {
	bw := &bitWriter{}
	bw.write(1, 1)
	bw.write(transformColorIndexing, 2)
	bw.write(uint32(len(pal)-1), 8)

	delta := make([]uint32, len(pal))
	delta[0] = pal[0]
	for i := 1; i < len(pal); i++ {
		delta[i] = subPixels(pal[i], pal[i-1])
	}
	encodeImage(bw, delta, len(pal), 1, false, e)

	packed, pw := packIndices(argb, w, h, pal)
	bw.write(0, 1)
	encodeImage(bw, packed, pw, h, true, e)
	return bw.bytes()
}

// encodeImage writes an entropy-coded image. Only the top-level image may
// use meta prefix codes.
func encodeImage(bw *bitWriter, pix []uint32, w, h int, top bool, e effort) {
	refs := backwardRefs(pix, w, e.iters)

	costs := cacheCosts(pix, refs, w)
	cacheBits := 0
	for b, c := range costs {
		if c < costs[cacheBits] {
			cacheBits = b
		}
	}
	applyCache(pix, refs, cacheBits)
	for i := range refs {
		if refs[i].kind == refCopy {
			refs[i].dist = planeCode(w, refs[i].dist)
		}
	}

	if cacheBits > 0 {
		bw.write(1, 1)
		bw.write(uint32(cacheBits), 4)
	} else {
		bw.write(0, 1)
	}
	if top {
		if bits, tileGroup, groups := clusterTiles(refs, w, h, cacheBits); groups != nil {
			bw.write(1, 1)
			writeGrouped(bw, refs, w, h, bits, tileGroup, groups, e)
			return
		}
		bw.write(0, 1)
	}

	hist := newHistogram(cacheBits)
	for _, r := range refs {
		hist.add(r)
	}
	codes := buildCodes(&hist)
	for i := range codes {
		writeHuffmanCode(bw, &codes[i])
	}
	for _, r := range refs {
		writeRef(bw, r, &codes)
	}
}

// writeGrouped writes the entropy image, one set of prefix codes per group
// and then refs, each coded with the group of its tile.
func writeGrouped(bw *bitWriter, refs []ref, w, h, bits int, tileGroup []uint32, groups []histogram, e effort) {
	bw.write(uint32(bits-2), 3)
	img := make([]uint32, len(tileGroup))
	for i, g := range tileGroup {
		img[i] = g << 8
	}
	encodeImage(bw, img, tiles(w, bits), tiles(h, bits), false, e)

	codes := make([]codeGroup, len(groups))
	for i := range groups {
		codes[i] = buildCodes(&groups[i])
		for j := range codes[i] {
			writeHuffmanCode(bw, &codes[i][j])
		}
	}
	forEachTile(refs, w, bits, func(tile int, r ref) {
		writeRef(bw, r, &codes[tileGroup[tile]])
	})
}

type codeGroup [5]huffmanCode

func buildCodes(h *histogram) codeGroup {
	return codeGroup{
		newHuffmanCode(h.green, 15),
		newHuffmanCode(h.red, 15),
		newHuffmanCode(h.blue, 15),
		newHuffmanCode(h.alpha, 15),
		newHuffmanCode(h.dist, 15),
	}
}

func writeRef(bw *bitWriter, r ref, g *codeGroup) {
	switch r.kind {
	case refLiteral:
		p := r.val
		g[0].put(bw, int(p>>8&0xff))
		g[1].put(bw, int(p>>16&0xff))
		g[2].put(bw, int(p&0xff))
		g[3].put(bw, int(p>>24))
	case refCache:
		g[0].put(bw, int(256+24+r.val))
	case refCopy:
		code, n := prefixEncode(r.val)
		g[0].put(bw, int(256+code))
		bw.write((r.val-1)&(1<<n-1), uint(n))
		code, n = prefixEncode(r.dist)
		g[4].put(bw, int(code))
		bw.write((r.dist-1)&(1<<n-1), uint(n))
	}
}
