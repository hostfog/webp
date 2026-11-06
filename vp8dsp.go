package webp

// Intra prediction modes. The first four double as 16x16 and chroma modes.
const (
	predDC = iota
	predTM
	predVE
	predHE
	predRD
	predVR
	predLD
	predVL
	predHD
	predHU
	numBModes
)

// workspace holds a block being reconstructed with its prediction context:
// row 0 is the row above (column 0 is the top-left pixel) and column 0 is the
// column to the left. Columns past the block hold the above-right pixels.
type workspace [17][21]uint8

func clip8(v int32) uint8 {
	return uint8(min(max(v, 0), 255))
}

func avg3(a, b, c uint8) uint8 {
	return uint8((int32(a) + 2*int32(b) + int32(c) + 2) >> 2)
}

func avg2(a, b uint8) uint8 {
	return uint8((int32(a) + int32(b) + 1) >> 1)
}

// predictBlock writes the n×n prediction for mode into ws at (y, x), with
// y and x at least 1. noTop and noLeft select the DC variants used at the
// image edges.
func predictBlock(ws *workspace, y, x, n, mode int, noTop, noLeft bool) {
	switch mode {
	case predDC:
		var sum, count uint32
		if !noTop {
			for i := 0; i < n; i++ {
				sum += uint32(ws[y-1][x+i])
			}
			count += uint32(n)
		}
		if !noLeft {
			for j := 0; j < n; j++ {
				sum += uint32(ws[y+j][x-1])
			}
			count += uint32(n)
		}
		v := uint8(0x80)
		if count > 0 {
			v = uint8((sum + count/2) / count)
		}
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				ws[y+j][x+i] = v
			}
		}
	case predTM:
		tl := int32(ws[y-1][x-1])
		for j := 0; j < n; j++ {
			l := int32(ws[y+j][x-1]) - tl
			for i := 0; i < n; i++ {
				ws[y+j][x+i] = clip8(l + int32(ws[y-1][x+i]))
			}
		}
	case predVE:
		for j := 0; j < n; j++ {
			copy(ws[y+j][x:x+n], ws[y-1][x:x+n])
		}
	case predHE:
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				ws[y+j][x+i] = ws[y+j][x-1]
			}
		}
	}
}

// predict4 writes the 4×4 subblock prediction for mode into ws at (y, x).
func predict4(ws *workspace, y, x, mode int) {
	t := ws[y-1][x-1 : x+8]
	tl, a, b, c, d, e, f, g, h := t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]
	p, q, r, s := ws[y][x-1], ws[y+1][x-1], ws[y+2][x-1], ws[y+3][x-1]
	set := func(rows [4][4]uint8) {
		for j := range rows {
			copy(ws[y+j][x:x+4], rows[j][:])
		}
	}

	switch mode {
	case predDC:
		sum := uint32(4)
		for i := 0; i < 4; i++ {
			sum += uint32(t[1+i]) + uint32(ws[y+i][x-1])
		}
		v := uint8(sum / 8)
		set([4][4]uint8{{v, v, v, v}, {v, v, v, v}, {v, v, v, v}, {v, v, v, v}})
	case predTM:
		predictBlock(ws, y, x, 4, predTM, false, false)
	case predVE:
		r0 := [4]uint8{avg3(tl, a, b), avg3(a, b, c), avg3(b, c, d), avg3(c, d, e)}
		set([4][4]uint8{r0, r0, r0, r0})
	case predHE:
		v0, v1, v2, v3 := avg3(tl, p, q), avg3(p, q, r), avg3(q, r, s), avg3(r, s, s)
		set([4][4]uint8{{v0, v0, v0, v0}, {v1, v1, v1, v1}, {v2, v2, v2, v2}, {v3, v3, v3, v3}})
	case predRD:
		srq, rqp, qpa := avg3(s, r, q), avg3(r, q, p), avg3(q, p, tl)
		pab, abc, bcd, cde := avg3(p, tl, a), avg3(tl, a, b), avg3(a, b, c), avg3(b, c, d)
		set([4][4]uint8{
			{pab, abc, bcd, cde},
			{qpa, pab, abc, bcd},
			{rqp, qpa, pab, abc},
			{srq, rqp, qpa, pab},
		})
	case predVR:
		ab, bc, cd, de := avg2(tl, a), avg2(a, b), avg2(b, c), avg2(c, d)
		rqp, qpa, pab := avg3(r, q, p), avg3(q, p, tl), avg3(p, tl, a)
		abc, bcd, cde := avg3(tl, a, b), avg3(a, b, c), avg3(b, c, d)
		set([4][4]uint8{
			{ab, bc, cd, de},
			{pab, abc, bcd, cde},
			{qpa, ab, bc, cd},
			{rqp, pab, abc, bcd},
		})
	case predLD:
		abc, bcd, cde, def := avg3(a, b, c), avg3(b, c, d), avg3(c, d, e), avg3(d, e, f)
		efg, fgh, ghh := avg3(e, f, g), avg3(f, g, h), avg3(g, h, h)
		set([4][4]uint8{
			{abc, bcd, cde, def},
			{bcd, cde, def, efg},
			{cde, def, efg, fgh},
			{def, efg, fgh, ghh},
		})
	case predVL:
		ab, bc, cd, de := avg2(a, b), avg2(b, c), avg2(c, d), avg2(d, e)
		abc, bcd, cde, def := avg3(a, b, c), avg3(b, c, d), avg3(c, d, e), avg3(d, e, f)
		efg, fgh := avg3(e, f, g), avg3(f, g, h)
		set([4][4]uint8{
			{ab, bc, cd, de},
			{abc, bcd, cde, def},
			{bc, cd, de, efg},
			{bcd, cde, def, fgh},
		})
	case predHD:
		sr, rq, qp, pa := avg2(s, r), avg2(r, q), avg2(q, p), avg2(p, tl)
		srq, rqp, qpa := avg3(s, r, q), avg3(r, q, p), avg3(q, p, tl)
		pab, abc, bcd := avg3(p, tl, a), avg3(tl, a, b), avg3(a, b, c)
		set([4][4]uint8{
			{pa, pab, abc, bcd},
			{qp, qpa, pa, pab},
			{rq, rqp, qp, qpa},
			{sr, srq, rq, rqp},
		})
	case predHU:
		pq, qr, rs := avg2(p, q), avg2(q, r), avg2(r, s)
		pqr, qrs, rss := avg3(p, q, r), avg3(q, r, s), avg3(r, s, s)
		set([4][4]uint8{
			{pq, pqr, qr, qrs},
			{qr, qrs, rs, rss},
			{rs, rss, s, s},
			{s, s, s, s},
		})
	}
}

// fdct computes the forward 4×4 DCT of src-pred.
func fdct(src, pred *[16]uint8, out *[16]int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		d0 := int32(src[4*i]) - int32(pred[4*i])
		d1 := int32(src[4*i+1]) - int32(pred[4*i+1])
		d2 := int32(src[4*i+2]) - int32(pred[4*i+2])
		d3 := int32(src[4*i+3]) - int32(pred[4*i+3])
		a0, a1, a2, a3 := d0+d3, d1+d2, d1-d2, d0-d3
		tmp[4*i] = (a0 + a1) * 8
		tmp[4*i+1] = (a2*2217 + a3*5352 + 1812) >> 9
		tmp[4*i+2] = (a0 - a1) * 8
		tmp[4*i+3] = (a3*2217 - a2*5352 + 937) >> 9
	}
	for i := 0; i < 4; i++ {
		a0, a1 := tmp[i]+tmp[12+i], tmp[4+i]+tmp[8+i]
		a2, a3 := tmp[4+i]-tmp[8+i], tmp[i]-tmp[12+i]
		out[i] = int16((a0 + a1 + 7) >> 4)
		out[4+i] = int16((a2*2217+a3*5352+12000)>>16 + b2i(a3 != 0))
		out[8+i] = int16((a0 - a1 + 7) >> 4)
		out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}

func b2i(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// idct adds the inverse DCT of in to the 4×4 block of ws at (y, x), exactly
// as a decoder does.
func idct(in *[16]int16, ws *workspace, y, x int) {
	const c1, c2 = 85627, 35468
	var m [4][4]int32
	for i := 0; i < 4; i++ {
		a := int32(in[i]) + int32(in[8+i])
		b := int32(in[i]) - int32(in[8+i])
		c := (int32(in[4+i])*c2)>>16 - (int32(in[12+i])*c1)>>16
		d := (int32(in[4+i])*c1)>>16 + (int32(in[12+i])*c2)>>16
		m[i] = [4]int32{a + d, b + c, b - c, a - d}
	}
	for j := 0; j < 4; j++ {
		dc := m[0][j] + 4
		a := dc + m[2][j]
		b := dc - m[2][j]
		c := (m[1][j]*c2)>>16 - (m[3][j]*c1)>>16
		d := (m[1][j]*c1)>>16 + (m[3][j]*c2)>>16
		row := ws[y+j][x : x+4]
		row[0] = clip8(int32(row[0]) + (a+d)>>3)
		row[1] = clip8(int32(row[1]) + (b+c)>>3)
		row[2] = clip8(int32(row[2]) + (b-c)>>3)
		row[3] = clip8(int32(row[3]) + (a-d)>>3)
	}
}

// fwht computes the Walsh-Hadamard transform of the 16 luma DC coefficients.
func fwht(in *[16]int16, out *[16]int16) {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		r := in[4*i : 4*i+4]
		a0, a1 := int32(r[0])+int32(r[2]), int32(r[1])+int32(r[3])
		a2, a3 := int32(r[1])-int32(r[3]), int32(r[0])-int32(r[2])
		tmp[4*i] = a0 + a1
		tmp[4*i+1] = a3 + a2
		tmp[4*i+2] = a3 - a2
		tmp[4*i+3] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0, a1 := tmp[i]+tmp[8+i], tmp[4+i]+tmp[12+i]
		a2, a3 := tmp[4+i]-tmp[12+i], tmp[i]-tmp[8+i]
		out[i] = int16((a0 + a1) >> 1)
		out[4+i] = int16((a3 + a2) >> 1)
		out[8+i] = int16((a3 - a2) >> 1)
		out[12+i] = int16((a0 - a1) >> 1)
	}
}

// iwht inverts fwht exactly as a decoder does.
func iwht(in *[16]int16, out *[16]int16) {
	var m [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(in[i]) + int32(in[12+i])
		a1 := int32(in[4+i]) + int32(in[8+i])
		a2 := int32(in[4+i]) - int32(in[8+i])
		a3 := int32(in[i]) - int32(in[12+i])
		m[i] = a0 + a1
		m[8+i] = a0 - a1
		m[4+i] = a3 + a2
		m[12+i] = a3 - a2
	}
	for i := 0; i < 4; i++ {
		dc := m[4*i] + 3
		a0 := dc + m[4*i+3]
		a1 := m[4*i+1] + m[4*i+2]
		a2 := m[4*i+1] - m[4*i+2]
		a3 := dc - m[4*i+3]
		out[4*i] = int16((a0 + a1) >> 3)
		out[4*i+1] = int16((a3 + a2) >> 3)
		out[4*i+2] = int16((a0 - a1) >> 3)
		out[4*i+3] = int16((a3 - a2) >> 3)
	}
}
