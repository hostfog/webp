package webp

import (
	"encoding/binary"
	"errors"
	"io"
	"math"
)

// quantizer holds the dequantization step and rounding bias of the DC and
// AC coefficients.
type quantizer struct {
	q    [2]int32
	bias [2]int32 // in 1/256 of a step
}

func (q *quantizer) quantize(in *[16]int16, out *[16]int16, first int) {
	for i := 0; i < 16; i++ {
		if i < first {
			out[i] = 0
			continue
		}
		k := min(i, 1)
		v := int32(in[i])
		neg := v < 0
		if neg {
			v = -v
		}
		l := min((v*256+q.bias[k]*q.q[k])/(q.q[k]*256), 2048)
		if neg {
			l = -l
		}
		out[i] = int16(l)
	}
}

func (q *quantizer) dequantize(in *[16]int16, out *[16]int16) {
	out[0] = int16(int32(in[0]) * q.q[0])
	for i := 1; i < 16; i++ {
		out[i] = int16(int32(in[i]) * q.q[1])
	}
}

// qualityToIndex maps quality to a quantizer index the way libwebp does.
func qualityToIndex(quality float32) int {
	c := float64(quality) / 100
	if c < 0.75 {
		c *= 2.0 / 3
	} else {
		c = 2*c - 1
	}
	return min(max(int(127*(1-math.Cbrt(c))), 0), 127)
}

// vp8Encoder encodes one VP8 key frame.
type vp8Encoder struct {
	width, height int
	mbw, mbh      int
	qi            int
	y1, y2, uv    quantizer
	lambda        float32
	filterLevel   int
	noSubblocks   bool // predict whole macroblocks only, to keep the mode partition small
	probs         tokenProbs

	y, u, v          []uint8 // source planes padded to whole macroblocks
	ry, ru, rv       []uint8 // reconstruction
	ystride, cstride int
}

func encodeLossy(w io.Writer, argb []uint32, width, height int, quality float32) error {
	e := newVP8Encoder(argb, width, height, quality)
	frame, err := e.encode()
	if err != nil {
		return err
	}

	hasAlpha := false
	for _, p := range argb {
		if p>>24 != 0xff {
			hasAlpha = true
			break
		}
	}
	if !hasAlpha {
		return writeRIFF(w, chunk{"VP8 ", frame})
	}

	alpha := make([]uint32, len(argb))
	for i, p := range argb {
		a := p >> 24
		alpha[i] = 0xff000000 | a<<16 | a<<8 | a
	}
	alph := append([]byte{1}, encodeLosslessBody(alpha, width, height, newEffort(quality))...)

	vp8x := make([]byte, 10)
	vp8x[0] = 0x10
	putUint24(vp8x[4:], uint32(width-1))
	putUint24(vp8x[7:], uint32(height-1))
	return writeRIFF(w, chunk{"VP8X", vp8x}, chunk{"ALPH", alph}, chunk{"VP8 ", frame})
}

func putUint24(b []byte, v uint32) {
	b[0], b[1], b[2] = byte(v), byte(v>>8), byte(v>>16)
}

func newVP8Encoder(argb []uint32, width, height int, quality float32) *vp8Encoder {
	e := &vp8Encoder{
		width:  width,
		height: height,
		mbw:    (width + 15) / 16,
		mbh:    (height + 15) / 16,
		probs:  defaultCoeffProbs,
	}
	e.ystride, e.cstride = 16*e.mbw, 8*e.mbw
	e.y = make([]uint8, e.ystride*16*e.mbh)
	e.u = make([]uint8, e.cstride*8*e.mbh)
	e.v = make([]uint8, e.cstride*8*e.mbh)
	e.ry = make([]uint8, len(e.y))
	e.ru = make([]uint8, len(e.u))
	e.rv = make([]uint8, len(e.v))
	e.toYUV(argb)

	qi := qualityToIndex(quality)
	e.qi = qi
	e.y1 = quantizer{q: [2]int32{int32(dcTable[qi]), int32(acTable[qi])}, bias: [2]int32{96, 110}}
	e.y2 = quantizer{q: [2]int32{2 * int32(dcTable[qi]), max(int32(acTable[qi])*155/100, 8)}, bias: [2]int32{96, 108}}
	e.uv = quantizer{q: [2]int32{int32(dcTable[min(qi, 117)]), int32(acTable[qi])}, bias: [2]int32{110, 115}}
	q := float32(acTable[qi])
	e.lambda = 0.03 * q * q
	e.filterLevel = min(int(q*0.3), 63)
	return e
}

// toYUV converts to BT.601 limited range YCbCr 4:2:0, replicating the edge
// pixels into the macroblock padding.
func (e *vp8Encoder) toYUV(argb []uint32) {
	w, h := e.width, e.height
	px := func(x, y int) (int32, int32, int32) {
		p := argb[min(y, h-1)*w+min(x, w-1)]
		return int32(p >> 16 & 0xff), int32(p >> 8 & 0xff), int32(p & 0xff)
	}
	for y := 0; y < 16*e.mbh; y++ {
		for x := 0; x < e.ystride; x++ {
			r, g, b := px(x, y)
			e.y[y*e.ystride+x] = uint8((16839*r + 33059*g + 6420*b + 16<<16 + 1<<15) >> 16)
		}
	}
	cw, ch := (w+1)/2, (h+1)/2
	for y := 0; y < 8*e.mbh; y++ {
		for x := 0; x < e.cstride; x++ {
			sx, sy := 2*min(x, cw-1), 2*min(y, ch-1)
			var r, g, b int32
			for _, d := range [4][2]int{{0, 0}, {1, 0}, {0, 1}, {1, 1}} {
				pr, pg, pb := px(sx+d[0], sy+d[1])
				r, g, b = r+pr, g+pg, b+pb
			}
			const round = 128<<18 + 1<<17
			e.u[y*e.cstride+x] = clip8((-9719*r - 19081*g + 28800*b + round) >> 18)
			e.v[y*e.cstride+x] = clip8((28800*r - 24116*g - 4684*b + round) >> 18)
		}
	}
}

// maxPartition0 is the size limit of the first partition, whose length is
// stored in 19 bits.
const maxPartition0 = 1<<19 - 1

func (e *vp8Encoder) encode() ([]byte, error) {
	mbs := make([]macroblock, e.mbw*e.mbh)
	top := make([]nzContext, e.mbw)
	topModes := make([][4]uint8, e.mbw)
	for mby := 0; mby < e.mbh; mby++ {
		var left nzContext
		var leftModes [4]uint8
		for mbx := 0; mbx < e.mbw; mbx++ {
			mb := &mbs[mby*e.mbw+mbx]
			e.analyze(mb, mbx, mby, &top[mbx], &left, &topModes[mbx], &leftModes)
		}
	}

	var stats tokenStats
	e.codeTokens(mbs, &tokenCoder{probs: &e.probs, stats: &stats})
	probs, updates := chooseProbs(&stats)

	tokens := newBoolEncoder()
	e.codeTokens(mbs, &tokenCoder{probs: &probs, enc: tokens})
	part2 := tokens.finish()

	part1 := e.header(mbs, &probs, &updates)
	if len(part1) > maxPartition0 {
		if e.noSubblocks {
			return nil, errors.New("webp: image too large for lossy encoding")
		}
		e.noSubblocks = true
		return e.encode()
	}

	out := make([]byte, 10, 10+len(part1)+len(part2))
	tag := uint32(len(part1))<<5 | 1<<4
	putUint24(out, tag)
	copy(out[3:], []byte{0x9d, 0x01, 0x2a})
	binary.LittleEndian.PutUint16(out[6:], uint16(e.width))
	binary.LittleEndian.PutUint16(out[8:], uint16(e.height))
	out = append(out, part1...)
	return append(out, part2...), nil
}

func (e *vp8Encoder) codeTokens(mbs []macroblock, t *tokenCoder) {
	top := make([]nzContext, e.mbw)
	for mby := 0; mby < e.mbh; mby++ {
		var left nzContext
		for mbx := 0; mbx < e.mbw; mbx++ {
			t.tokens(&mbs[mby*e.mbw+mbx], &top[mbx], &left)
		}
	}
}

func (e *vp8Encoder) header(mbs []macroblock, probs *tokenProbs, updated *probFlags) []byte {
	b := newBoolEncoder()
	b.putLiteral(0, 1) // color space
	b.putLiteral(0, 1) // clamping required
	b.putLiteral(0, 1) // no segmentation

	b.putLiteral(0, 1) // normal loop filter
	b.putLiteral(uint32(e.filterLevel), 6)
	b.putLiteral(0, 3) // sharpness
	b.putLiteral(0, 1) // no filter deltas

	b.putLiteral(0, 2) // one token partition

	b.putLiteral(uint32(e.qi), 7)
	b.putLiteral(0, 5) // no quantizer deltas
	b.putLiteral(0, 1) // refresh entropy probs

	for i := range probs {
		for j := range probs[i] {
			for k := range probs[i][j] {
				for l, p := range probs[i][j][k] {
					up := updated[i][j][k][l]
					b.putBit(up, coeffUpdateProbs[i][j][k][l])
					if up {
						b.putLiteral(uint32(p), 8)
					}
				}
			}
		}
	}

	skipped := 0
	for i := range mbs {
		if mbs[i].skip {
			skipped++
		}
	}
	skipProb := uint8(0)
	b.putBit(skipped > 0, 128)
	if skipped > 0 {
		skipProb = uint8(min(max((len(mbs)-skipped)*256/len(mbs), 1), 255))
		b.putLiteral(uint32(skipProb), 8)
	}

	topModes := make([][4]uint8, e.mbw)
	for mby := 0; mby < e.mbh; mby++ {
		var leftModes [4]uint8
		for mbx := 0; mbx < e.mbw; mbx++ {
			mb := &mbs[mby*e.mbw+mbx]
			if skipped > 0 {
				b.putBit(mb.skip, skipProb)
			}
			var c modeCost
			c.enc = b
			c.modes(mb, &topModes[mbx], &leftModes)
		}
	}
	return b.finish()
}

// context fills row 0 and column 0 of ws with the reconstructed pixels
// around the n×n block of macroblock (mbx, mby), using 127 above and 129 to
// the left of the image as decoders do.
func (e *vp8Encoder) context(ws *workspace, rec []uint8, stride, n, mbx, mby int) {
	x0, y0 := n*mbx, n*mby
	if mby == 0 {
		for i := 0; i <= n+4; i++ {
			ws[0][i] = 127
		}
	} else {
		row := rec[(y0-1)*stride:]
		copy(ws[0][1:n+1], row[x0:x0+n])
		if n == 16 {
			for i := 0; i < 4; i++ {
				if mbx == e.mbw-1 {
					ws[0][17+i] = row[x0+15]
				} else {
					ws[0][17+i] = row[x0+16+i]
				}
			}
		}
		if mbx == 0 {
			ws[0][0] = 129
		} else {
			ws[0][0] = row[x0-1]
		}
	}
	for j := 0; j < n; j++ {
		if mbx == 0 {
			ws[1+j][0] = 129
		} else {
			ws[1+j][0] = rec[(y0+j)*stride+x0-1]
		}
	}
}

func blockOf(ws *workspace, y, x int) (b [16]uint8) {
	for j := 0; j < 4; j++ {
		copy(b[4*j:4*j+4], ws[y+j][x:x+4])
	}
	return b
}

func setBlock(ws *workspace, y, x int, b *[16]uint8) {
	for j := 0; j < 4; j++ {
		copy(ws[y+j][x:x+4], b[4*j:4*j+4])
	}
}

func sse(a, b *[16]uint8) float32 {
	var s int32
	for i := range a {
		d := int32(a[i]) - int32(b[i])
		s += d * d
	}
	return float32(s)
}

// sourceBlocks returns the 4×4 blocks of an n×n area of a plane in raster
// order.
func sourceBlocks(plane []uint8, stride, n, mbx, mby int) [][16]uint8 {
	nb := n / 4
	out := make([][16]uint8, nb*nb)
	for k := range out {
		y, x := n*mby+4*(k/nb), n*mbx+4*(k%nb)
		for j := 0; j < 4; j++ {
			copy(out[k][4*j:4*j+4], plane[(y+j)*stride+x:])
		}
	}
	return out
}

// analyze chooses the prediction modes of a macroblock by rate-distortion
// cost, quantizes it and updates the reconstruction and contexts.
func (e *vp8Encoder) analyze(mb *macroblock, mbx, mby int, top, left *nzContext, topModes, leftModes *[4]uint8) {
	src := sourceBlocks(e.y, e.ystride, 16, mbx, mby)
	var ws workspace
	e.context(&ws, e.ry, e.ystride, 16, mbx, mby)

	score, rec := e.analyzeLuma16(mb, &ws, src, mbx, mby, top, left)
	if !e.noSubblocks {
		var mb4 macroblock
		if score4, rec4 := e.analyzeLuma4(&mb4, &ws, src, score, top, left, topModes, leftModes); score4 < score {
			*mb, rec = mb4, rec4
		}
	}
	for j := 0; j < 16; j++ {
		copy(e.ry[(16*mby+j)*e.ystride+16*mbx:][:16], rec[1+j][1:17])
	}
	e.analyzeChroma(mb, mbx, mby, top, left)

	mb.skip = !mb.i16 || allZero(&mb.y2)
	for k := range mb.y {
		mb.skip = mb.skip && allZero(&mb.y[k])
	}
	for k := range mb.u {
		mb.skip = mb.skip && allZero(&mb.u[k]) && allZero(&mb.v[k])
	}

	tc := tokenCoder{probs: &e.probs}
	tc.tokens(mb, top, left)
	var mc modeCost
	mc.modes(mb, topModes, leftModes)
}

// analyzeLuma16 tries the four whole-macroblock luma predictors.
func (e *vp8Encoder) analyzeLuma16(mb *macroblock, ws *workspace, src [][16]uint8, mbx, mby int, top, left *nzContext) (float32, workspace) {
	best := float32(math.MaxFloat32)
	var bestRec workspace
	for _, mode := range []uint8{predDC, predTM, predVE, predHE} {
		rec := *ws
		m := macroblock{i16: true, ymode: mode}
		predictBlock(&rec, 1, 1, 16, int(mode), mby == 0, mbx == 0)

		var coeffs [16][16]int16
		var dc, wht, dq [16]int16
		for k := range src {
			pred := blockOf(&rec, 1+4*(k/4), 1+4*(k%4))
			fdct(&src[k], &pred, &coeffs[k])
			dc[k] = coeffs[k][0]
		}
		fwht(&dc, &wht)
		e.y2.quantize(&wht, &m.y2, 0)
		e.y2.dequantize(&m.y2, &dq)
		iwht(&dq, &dc)

		var dist float32
		for k := range src {
			e.y1.quantize(&coeffs[k], &m.y[k], 1)
			var c [16]int16
			e.y1.dequantize(&m.y[k], &c)
			c[0] = dc[k]
			y, x := 1+4*(k/4), 1+4*(k%4)
			idct(&c, &rec, y, x)
			b := blockOf(&rec, y, x)
			dist += sse(&b, &src[k])
		}

		tc := tokenCoder{probs: &e.probs}
		t, l := *top, *left
		nz := tc.block(&m.y2, planeY2, int(t.y2+l.y2), 0)
		t.y2, l.y2 = nz, nz
		tc.lumaTokens(&m.y, planeY1WithY2, 1, &t, &l)
		var mc modeCost
		mc.ymode(mode)

		if score := dist + e.lambda*(tc.bits+mc.bits); score < best {
			best, bestRec, *mb = score, rec, m
		}
	}
	return best, bestRec
}

// analyzeLuma4 chooses a predictor for each 4×4 subblock in turn. It gives
// up once the cost reaches limit.
func (e *vp8Encoder) analyzeLuma4(mb *macroblock, ws *workspace, src [][16]uint8, limit float32, top, left *nzContext, topModes, leftModes *[4]uint8) (float32, workspace) {
	rec := *ws
	for r := 4; r < 16; r += 4 {
		copy(rec[r][17:21], rec[0][17:21])
	}
	var mc modeCost
	mc.put(false, 145)
	total := e.lambda * mc.bits
	t, l := *top, *left
	tm, lm := *topModes, *leftModes
	for k := 0; k < 16 && total < limit; k++ {
		j, i := k/4, k%4
		y, x := 1+4*j, 1+4*i
		best := float32(math.MaxFloat32)
		var bestBlock [16]uint8
		var bestNZ uint8
		for mode := uint8(0); mode < numBModes; mode++ {
			predict4(&rec, y, x, int(mode))
			pred := blockOf(&rec, y, x)
			var coeffs, levels, dq [16]int16
			fdct(&src[k], &pred, &coeffs)
			e.y1.quantize(&coeffs, &levels, 0)
			e.y1.dequantize(&levels, &dq)
			idct(&dq, &rec, y, x)
			b := blockOf(&rec, y, x)

			tc := tokenCoder{probs: &e.probs}
			nz := tc.block(&levels, planeY1, int(t.y[i]+l.y[j]), 0)
			var mc modeCost
			mc.bmode(mode, tm[i], lm[j])
			if score := sse(&b, &src[k]) + e.lambda*(tc.bits+mc.bits); score < best {
				best, bestBlock, bestNZ = score, b, nz
				mb.bmodes[k], mb.y[k] = mode, levels
			}
		}
		setBlock(&rec, y, x, &bestBlock)
		t.y[i], l.y[j] = bestNZ, bestNZ
		tm[i], lm[j] = mb.bmodes[k], mb.bmodes[k]
		total += best
	}
	return total, rec
}

func (e *vp8Encoder) analyzeChroma(mb *macroblock, mbx, mby int, top, left *nzContext) {
	srcU := sourceBlocks(e.u, e.cstride, 8, mbx, mby)
	srcV := sourceBlocks(e.v, e.cstride, 8, mbx, mby)
	var wsU, wsV workspace
	e.context(&wsU, e.ru, e.cstride, 8, mbx, mby)
	e.context(&wsV, e.rv, e.cstride, 8, mbx, mby)

	best := float32(math.MaxFloat32)
	var bestU, bestV workspace
	for _, mode := range []uint8{predDC, predTM, predVE, predHE} {
		cu, cv := wsU, wsV
		var lu, lv [4][16]int16
		dist := e.codeChroma(&cu, srcU, &lu, int(mode), mbx, mby) + e.codeChroma(&cv, srcV, &lv, int(mode), mbx, mby)

		tc := tokenCoder{probs: &e.probs}
		t, l := *top, *left
		tc.chromaTokens(&lu, &t.u, &l.u)
		tc.chromaTokens(&lv, &t.v, &l.v)
		var mc modeCost
		mc.uvmode(mode)
		if score := dist + e.lambda*(tc.bits+mc.bits); score < best {
			best, bestU, bestV = score, cu, cv
			mb.uvmode, mb.u, mb.v = mode, lu, lv
		}
	}
	for j := 0; j < 8; j++ {
		copy(e.ru[(8*mby+j)*e.cstride+8*mbx:][:8], bestU[1+j][1:9])
		copy(e.rv[(8*mby+j)*e.cstride+8*mbx:][:8], bestV[1+j][1:9])
	}
}

func (e *vp8Encoder) codeChroma(ws *workspace, src [][16]uint8, levels *[4][16]int16, mode, mbx, mby int) float32 {
	predictBlock(ws, 1, 1, 8, mode, mby == 0, mbx == 0)
	var dist float32
	for k := range src {
		y, x := 1+4*(k/2), 1+4*(k%2)
		pred := blockOf(ws, y, x)
		var coeffs, dq [16]int16
		fdct(&src[k], &pred, &coeffs)
		e.uv.quantize(&coeffs, &levels[k], 0)
		e.uv.dequantize(&levels[k], &dq)
		idct(&dq, ws, y, x)
		rec := blockOf(ws, y, x)
		dist += sse(&rec, &src[k])
	}
	return dist
}

func allZero(b *[16]int16) bool {
	return *b == [16]int16{}
}

// smoothTransparent replaces the color of fully transparent pixels with the
// mean color of the visible pixels in their 8×8 block, or with the fill of
// a neighbouring block when the whole block is transparent. Flat hidden
// areas cost almost nothing to code and do not bleed into visible ones.
func smoothTransparent(argb []uint32, w, h int) {
	const size = 8
	above := make([]uint32, tiles(w, 3))
	for by := 0; by < h; by += size {
		var left uint32
		for bx, i := 0, 0; bx < w; bx, i = bx+size, i+1 {
			var r, g, b, n uint32
			for y := by; y < min(by+size, h); y++ {
				for _, p := range argb[y*w+bx : y*w+min(bx+size, w)] {
					if p>>24 != 0 {
						r, g, b, n = r+p>>16&0xff, g+p>>8&0xff, b+p&0xff, n+1
					}
				}
			}
			fill := left
			switch {
			case n > 0:
				fill = (r+n/2)/n<<16 | (g+n/2)/n<<8 | (b+n/2)/n
			case bx == 0:
				fill = above[i]
			}
			left, above[i] = fill, fill
			for y := by; y < min(by+size, h); y++ {
				row := argb[y*w+bx : y*w+min(bx+size, w)]
				for k, p := range row {
					if p>>24 == 0 {
						row[k] = fill
					}
				}
			}
		}
	}
}
