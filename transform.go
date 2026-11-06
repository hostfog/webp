package webp

import (
	"math"
	"slices"
)

const (
	transformPredictor = iota
	transformCrossColor
	transformSubtractGreen
	transformColorIndexing
)

func tiles(size, bits int) int {
	return (size + 1<<bits - 1) >> bits
}

func subtractGreen(pix []uint32) {
	for i, p := range pix {
		g := p >> 8 & 0xff
		rb := (p&0x00ff00ff | 0xff00ff00) - (g<<16 | g)
		pix[i] = p&0xff00ff00 | rb&0x00ff00ff
	}
}

func subPixels(a, b uint32) uint32 {
	ag := 0x00ff00ff + a&0xff00ff00 - b&0xff00ff00
	rb := 0xff00ff00 + a&0x00ff00ff - b&0x00ff00ff
	return ag&0xff00ff00 | rb&0x00ff00ff
}

func average2(a, b uint32) uint32 {
	return (a^b)&0xfefefefe>>1 + a&b
}

func channels(p uint32) [4]int32 {
	return [4]int32{int32(p >> 24), int32(p >> 16 & 0xff), int32(p >> 8 & 0xff), int32(p & 0xff)}
}

func pack(c [4]int32) uint32 {
	return uint32(c[0])<<24 | uint32(c[1])<<16 | uint32(c[2])<<8 | uint32(c[3])
}

func clamp255(v int32) int32 {
	return min(max(v, 0), 255)
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

func selectPred(l, t, tl uint32) uint32 {
	cl, ct, ctl := channels(l), channels(t), channels(tl)
	var pl, pt int32
	for i := range cl {
		pl += abs32(ctl[i] - ct[i])
		pt += abs32(ctl[i] - cl[i])
	}
	if pl < pt {
		return l
	}
	return t
}

func clampAddSubtractFull(a, b, c uint32) uint32 {
	ca, cb, cc := channels(a), channels(b), channels(c)
	var out [4]int32
	for i := range out {
		out[i] = clamp255(ca[i] + cb[i] - cc[i])
	}
	return pack(out)
}

func clampAddSubtractHalf(a, b uint32) uint32 {
	ca, cb := channels(a), channels(b)
	var out [4]int32
	for i := range out {
		out[i] = clamp255(ca[i] + (ca[i]-cb[i])/2)
	}
	return pack(out)
}

func predictPixel(mode int, l, t, tr, tl uint32) uint32 {
	switch mode {
	case 0:
		return 0xff000000
	case 1:
		return l
	case 2:
		return t
	case 3:
		return tr
	case 4:
		return tl
	case 5:
		return average2(average2(l, tr), t)
	case 6:
		return average2(l, tl)
	case 7:
		return average2(l, t)
	case 8:
		return average2(tl, t)
	case 9:
		return average2(t, tr)
	case 10:
		return average2(average2(l, tl), average2(t, tr))
	case 11:
		return selectPred(l, t, tl)
	case 12:
		return clampAddSubtractFull(l, t, tl)
	default:
		return clampAddSubtractHalf(average2(l, t), tl)
	}
}

// residualCost approximates the bits needed for a residual byte, which is
// cheap when close to zero.
var residualCost = func() [256]float32 {
	var t [256]float32
	for i := range t {
		t[i] = float32(math.Log2(1 + math.Abs(float64(int8(i)))))
	}
	return t
}()

func pixelCost(p uint32) float32 {
	return residualCost[p>>24] + residualCost[p>>16&0xff] + residualCost[p>>8&0xff] + residualCost[p&0xff]
}

// predict chooses a predictor for each tile and returns the residuals and
// the predictor sub-image.
func predict(pix []uint32, w, h, bits int) (res, modes []uint32) {
	tw, th := tiles(w, bits), tiles(h, bits)
	res = make([]uint32, len(pix))
	modes = make([]uint32, tw*th)
	size := 1 << bits

	for ty := 0; ty < th; ty++ {
		for tx := 0; tx < tw; tx++ {
			x0, y0 := tx*size, ty*size
			x1, y1 := min(x0+size, w), min(y0+size, h)
			best, bestCost := 0, float32(math.MaxFloat32)
			for mode := 0; mode < 14; mode++ {
				var cost float32
				for y := max(y0, 1); y < y1 && cost < bestCost; y++ {
					for x := max(x0, 1); x < x1; x++ {
						i := y*w + x
						p := predictPixel(mode, pix[i-1], pix[i-w], pix[i-w+1], pix[i-w-1])
						cost += pixelCost(subPixels(pix[i], p))
					}
				}
				if cost < bestCost {
					best, bestCost = mode, cost
				}
			}
			modes[ty*tw+tx] = 0xff000000 | uint32(best)<<8
		}
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			var p uint32
			switch {
			case y == 0 && x == 0:
				p = 0xff000000
			case y == 0:
				p = pix[i-1]
			case x == 0:
				p = pix[i-w]
			default:
				mode := int(modes[(y>>bits)*tw+x>>bits] >> 8 & 0xff)
				p = predictPixel(mode, pix[i-1], pix[i-w], pix[i-w+1], pix[i-w-1])
			}
			res[i] = subPixels(pix[i], p)
		}
	}
	return res, modes
}

func colorDelta(t, c int8) int32 {
	return int32(t) * int32(c) >> 5
}

type crossColorElem struct {
	greenToRed, greenToBlue, redToBlue int8
}

// crossColor decorrelates red and blue from green in place, choosing the
// multipliers per tile, and returns the cross-color sub-image.
func crossColor(pix []uint32, w, h, bits int) []uint32 {
	tw, th := tiles(w, bits), tiles(h, bits)
	out := make([]uint32, tw*th)
	size := 1 << bits
	var prev crossColorElem

	tile := make([]uint32, 0, size*size)
	for ty := 0; ty < th; ty++ {
		for tx := 0; tx < tw; tx++ {
			x0, y0 := tx*size, ty*size
			x1, y1 := min(x0+size, w), min(y0+size, h)
			tile = tile[:0]
			for y := y0; y < y1; y++ {
				tile = append(tile, pix[y*w+x0:y*w+x1]...)
			}

			e := chooseCrossColor(tile, prev)
			if ty > 0 {
				above := unpackCrossColor(out[(ty-1)*tw+tx])
				if crossColorCost(tile, above) <= crossColorCost(tile, e) {
					e = above
				}
			}
			prev = e
			out[ty*tw+tx] = 0xff000000 | uint32(uint8(e.redToBlue))<<16 | uint32(uint8(e.greenToBlue))<<8 | uint32(uint8(e.greenToRed))

			for y := y0; y < y1; y++ {
				row := pix[y*w+x0 : y*w+x1]
				for i, p := range row {
					row[i] = applyCrossColor(p, e)
				}
			}
		}
	}
	return out
}

func unpackCrossColor(p uint32) crossColorElem {
	return crossColorElem{greenToRed: int8(p), greenToBlue: int8(p >> 8), redToBlue: int8(p >> 16)}
}

func applyCrossColor(p uint32, e crossColorElem) uint32 {
	r, g, b := int8(p>>16), int8(p>>8), int8(p)
	nr := int32(r) - colorDelta(e.greenToRed, g)
	nb := int32(b) - colorDelta(e.greenToBlue, g) - colorDelta(e.redToBlue, r)
	return p&0xff00ff00 | uint32(uint8(nr))<<16 | uint32(uint8(nb))
}

func crossColorCost(tile []uint32, e crossColorElem) float32 {
	var cost float32
	for _, p := range tile {
		q := applyCrossColor(p, e)
		cost += residualCost[q>>16&0xff] + residualCost[q&0xff]
	}
	return cost
}

// chooseCrossColor fits the multipliers by least squares, then refines them
// with a small search on the actual residual cost.
func chooseCrossColor(tile []uint32, prev crossColorElem) crossColorElem {
	var gg, rg, rr, bg, br float64
	for _, p := range tile {
		r, g, b := float64(int8(p>>16)), float64(int8(p>>8)), float64(int8(p))
		gg += g * g
		rg += r * g
		rr += r * r
		bg += b * g
		br += b * r
	}
	toInt8 := func(v float64) int8 {
		if math.IsNaN(v) {
			return 0
		}
		return int8(min(max(math.Round(v), -128), 127))
	}

	var e crossColorElem
	if gg > 0 {
		e.greenToRed = toInt8(32 * rg / gg)
	}
	if det := gg*rr - rg*rg; det > 0 {
		e.greenToBlue = toInt8(32 * (bg*rr - br*rg) / det)
		e.redToBlue = toInt8(32 * (br*gg - bg*rg) / det)
	} else if gg > 0 {
		e.greenToBlue = toInt8(32 * bg / gg)
	}

	best := crossColorCost(tile, e)
	try := func(c crossColorElem) {
		if cost := crossColorCost(tile, c); cost < best {
			e, best = c, cost
		}
	}
	try(crossColorElem{})
	try(prev)
	for _, d := range []int8{-2, -1, 1, 2} {
		c := e
		c.greenToRed += d
		try(c)
	}
	for _, d := range []int8{-2, -1, 1, 2} {
		c := e
		c.greenToBlue += d
		try(c)
	}
	for _, d := range []int8{-2, -1, 1, 2} {
		c := e
		c.redToBlue += d
		try(c)
	}
	return e
}

// buildPalette returns the sorted distinct colors of pix if there are at
// most 256.
func buildPalette(pix []uint32) ([]uint32, bool) {
	seen := make(map[uint32]struct{})
	last := ^pix[0]
	for _, p := range pix {
		if p == last {
			continue
		}
		last = p
		if _, ok := seen[p]; !ok {
			if len(seen) == 256 {
				return nil, false
			}
			seen[p] = struct{}{}
		}
	}
	pal := make([]uint32, 0, len(seen))
	for p := range seen {
		pal = append(pal, p)
	}
	slices.Sort(pal)
	return pal, true
}

// packIndices replaces each pixel by its palette index in the green channel,
// bundling several indices per pixel when the palette is small.
func packIndices(pix []uint32, w, h int, pal []uint32) ([]uint32, int) {
	index := make(map[uint32]uint32, len(pal))
	for i, p := range pal {
		index[p] = uint32(i)
	}
	var xbits int
	switch {
	case len(pal) <= 2:
		xbits = 3
	case len(pal) <= 4:
		xbits = 2
	case len(pal) <= 16:
		xbits = 1
	}
	pw := tiles(w, xbits)
	shift := uint(8 >> xbits)
	mask := 1<<xbits - 1
	out := make([]uint32, pw*h)

	last, lastIdx := pix[0], index[pix[0]]
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p := pix[y*w+x]
			if p != last {
				last, lastIdx = p, index[p]
			}
			out[y*pw+x>>xbits] |= lastIdx << (8 + shift*uint(x&mask))
		}
	}
	for i := range out {
		out[i] |= 0xff000000
	}
	return out, pw
}
