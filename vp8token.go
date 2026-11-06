package webp

import "math"

const (
	numPlanes   = 4
	numBands    = 8
	numContexts = 3
	numProbs    = 11
)

const (
	planeY1WithY2 = iota
	planeY2
	planeUV
	planeY1
)

var (
	zigzag   = [16]uint8{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}
	bands    = [17]uint8{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0}
	catProbs = [4][]uint8{
		{173, 148, 140},
		{176, 155, 140, 135},
		{180, 157, 141, 134, 130},
		{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129},
	}
)

type tokenProbs [numPlanes][numBands][numContexts][numProbs]uint8

type tokenStats [numPlanes][numBands][numContexts][numProbs][2]uint32

type probFlags [numPlanes][numBands][numContexts][numProbs]bool

// bitCosts[n] is the cost in bits of an event with probability n/256.
var bitCosts = func() [257]float32 {
	var t [257]float32
	for i := 1; i < len(t); i++ {
		t[i] = float32(-math.Log2(float64(i) / 256))
	}
	return t
}()

func bitCost(b bool, prob uint8) float32 {
	if b {
		return bitCosts[256-int(prob)]
	}
	return bitCosts[prob]
}

// tokenCoder codes coefficient tokens. It always accumulates the cost in
// bits, and also writes to enc or counts into stats when they are set.
type tokenCoder struct {
	probs *tokenProbs
	enc   *boolEncoder
	stats *tokenStats
	bits  float32
}

func (t *tokenCoder) put(b bool, plane int, band uint8, ctx, i int) {
	p := t.probs[plane][band][ctx][i]
	if t.enc != nil {
		t.enc.putBit(b, p)
	}
	if t.stats != nil {
		t.stats[plane][band][ctx][i][b2i(b)]++
	}
	t.bits += bitCost(b, p)
}

func (t *tokenCoder) putFixed(b bool, prob uint8) {
	if t.enc != nil {
		t.enc.putBit(b, prob)
	}
	t.bits += bitCost(b, prob)
}

// block codes the levels of a 4×4 block from position first in zigzag order
// and reports whether any of them is non-zero.
func (t *tokenCoder) block(levels *[16]int16, plane, ctx, first int) uint8 {
	last := -1
	for n := 15; n >= first; n-- {
		if levels[zigzag[n]] != 0 {
			last = n
			break
		}
	}
	if last < 0 {
		t.put(false, plane, bands[first], ctx, 0)
		return 0
	}
	t.put(true, plane, bands[first], ctx, 0)

	for n := first; n <= last; n++ {
		band := bands[n]
		l := levels[zigzag[n]]
		v := int(l)
		if v < 0 {
			v = -v
		}
		if v == 0 {
			t.put(false, plane, band, ctx, 1)
			ctx = 0
			continue
		}
		t.put(true, plane, band, ctx, 1)
		if v == 1 {
			t.put(false, plane, band, ctx, 2)
			ctx = 1
		} else {
			t.put(true, plane, band, ctx, 2)
			t.putLarge(v, plane, band, ctx)
			ctx = 2
		}
		t.putFixed(l < 0, 128)
		if n == 15 {
			break
		}
		t.put(n < last, plane, bands[n+1], ctx, 0)
	}
	return 1
}

func (t *tokenCoder) putLarge(v, plane int, band uint8, ctx int) {
	switch {
	case v <= 4:
		t.put(false, plane, band, ctx, 3)
		if v == 2 {
			t.put(false, plane, band, ctx, 4)
		} else {
			t.put(true, plane, band, ctx, 4)
			t.put(v == 4, plane, band, ctx, 5)
		}
	case v <= 10:
		t.put(true, plane, band, ctx, 3)
		t.put(false, plane, band, ctx, 6)
		if v <= 6 {
			t.put(false, plane, band, ctx, 7)
			t.putFixed(v == 6, 159)
		} else {
			t.put(true, plane, band, ctx, 7)
			t.putFixed((v-7)&2 != 0, 165)
			t.putFixed((v-7)&1 != 0, 145)
		}
	default:
		t.put(true, plane, band, ctx, 3)
		t.put(true, plane, band, ctx, 6)
		cat := 0
		for cat < 3 && v >= 3+(16<<cat) {
			cat++
		}
		t.put(cat >= 2, plane, band, ctx, 8)
		t.put(cat&1 == 1, plane, band, ctx, 9+cat>>1)
		extra := v - (3 + 8<<cat)
		probs := catProbs[cat]
		for i, p := range probs {
			t.putFixed(extra>>(len(probs)-1-i)&1 == 1, p)
		}
	}
}

// nzContext records which neighbouring blocks had non-zero coefficients:
// per column for the row above, per row for the column to the left.
type nzContext struct {
	y    [4]uint8
	u, v [2]uint8
	y2   uint8
}

type macroblock struct {
	i16    bool
	ymode  uint8
	bmodes [16]uint8
	uvmode uint8
	skip   bool
	y2     [16]int16
	y      [16][16]int16
	u, v   [4][16]int16
}

// tokens codes all coefficients of mb and updates the contexts.
func (t *tokenCoder) tokens(mb *macroblock, top, left *nzContext) {
	if mb.skip {
		if mb.i16 {
			top.y2, left.y2 = 0, 0
		}
		top.y, left.y = [4]uint8{}, [4]uint8{}
		top.u, left.u = [2]uint8{}, [2]uint8{}
		top.v, left.v = [2]uint8{}, [2]uint8{}
		return
	}
	plane, first := planeY1, 0
	if mb.i16 {
		nz := t.block(&mb.y2, planeY2, int(top.y2+left.y2), 0)
		top.y2, left.y2 = nz, nz
		plane, first = planeY1WithY2, 1
	}
	t.lumaTokens(&mb.y, plane, first, top, left)
	t.chromaTokens(&mb.u, &top.u, &left.u)
	t.chromaTokens(&mb.v, &top.v, &left.v)
}

func (t *tokenCoder) lumaTokens(y *[16][16]int16, plane, first int, top, left *nzContext) {
	for j := 0; j < 4; j++ {
		for i := 0; i < 4; i++ {
			nz := t.block(&y[4*j+i], plane, int(top.y[i]+left.y[j]), first)
			top.y[i], left.y[j] = nz, nz
		}
	}
}

func (t *tokenCoder) chromaTokens(c *[4][16]int16, top, left *[2]uint8) {
	for j := 0; j < 2; j++ {
		for i := 0; i < 2; i++ {
			nz := t.block(&c[2*j+i], planeUV, int(top[i]+left[j]), 0)
			top[i], left[j] = nz, nz
		}
	}
}

// chooseProbs updates each token probability whose savings outweigh the
// cost of signalling it.
func chooseProbs(stats *tokenStats) (probs tokenProbs, updated probFlags) {
	probs = defaultCoeffProbs
	for i := range stats {
		for j := range stats[i] {
			for k := range stats[i][j] {
				for l, s := range stats[i][j][k] {
					n0, n1 := s[0], s[1]
					if n0 == 0 && n1 == 0 {
						continue
					}
					total := uint64(n0) + uint64(n1)
					p := uint8(min(max((uint64(n0)*256+total/2)/total, 1), 255))
					old := probs[i][j][k][l]
					cost := func(p uint8) float32 {
						return float32(n0)*bitCost(false, p) + float32(n1)*bitCost(true, p)
					}
					up := coeffUpdateProbs[i][j][k][l]
					saving := cost(old) - cost(p) - 8 - bitCost(true, up) + bitCost(false, up)
					if saving > 0 {
						probs[i][j][k][l] = p
						updated[i][j][k][l] = true
					}
				}
			}
		}
	}
	return probs, updated
}

// modeCost codes macroblock prediction modes, accumulating their cost and
// writing them when enc is set.
type modeCost struct {
	enc  *boolEncoder
	bits float32
}

func (c *modeCost) put(b bool, prob uint8) {
	if c.enc != nil {
		c.enc.putBit(b, prob)
	}
	c.bits += bitCost(b, prob)
}

func (c *modeCost) ymode(mode uint8) {
	c.put(true, 145)
	switch mode {
	case predDC:
		c.put(false, 156)
		c.put(false, 163)
	case predVE:
		c.put(false, 156)
		c.put(true, 163)
	case predHE:
		c.put(true, 156)
		c.put(false, 128)
	case predTM:
		c.put(true, 156)
		c.put(true, 128)
	}
}

func (c *modeCost) bmode(mode, above, left uint8) {
	p := &bModeProbs[above][left]
	c.put(mode != predDC, p[0])
	if mode == predDC {
		return
	}
	c.put(mode != predTM, p[1])
	if mode == predTM {
		return
	}
	c.put(mode != predVE, p[2])
	if mode == predVE {
		return
	}
	switch mode {
	case predHE, predRD, predVR:
		c.put(false, p[3])
		c.put(mode != predHE, p[4])
		if mode != predHE {
			c.put(mode == predVR, p[5])
		}
	default:
		c.put(true, p[3])
		c.put(mode != predLD, p[6])
		if mode == predLD {
			return
		}
		c.put(mode != predVL, p[7])
		if mode != predVL {
			c.put(mode == predHU, p[8])
		}
	}
}

func (c *modeCost) uvmode(mode uint8) {
	c.put(mode != predDC, 142)
	if mode == predDC {
		return
	}
	c.put(mode != predVE, 114)
	if mode != predVE {
		c.put(mode == predTM, 183)
	}
}

// modes codes the prediction modes of mb and updates the subblock mode
// contexts.
func (c *modeCost) modes(mb *macroblock, top, left *[4]uint8) {
	if mb.i16 {
		c.ymode(mb.ymode)
		*top = [4]uint8{mb.ymode, mb.ymode, mb.ymode, mb.ymode}
		*left = *top
	} else {
		c.put(false, 145)
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				m := mb.bmodes[4*j+i]
				c.bmode(m, top[i], left[j])
				top[i], left[j] = m, m
			}
		}
	}
	c.uvmode(mb.uvmode)
}
