package webp

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"testing"

	"golang.org/x/image/vp8"
)

// psnr compares the RGB channels of two images, decoding YCbCr with the
// BT.601 limited range conversion that WebP specifies.
func psnr(want, got image.Image) float64 {
	b := want.Bounds()
	var sum float64
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			wr, wg, wb := rgbAt(want, x, y)
			gr, gg, gb := rgbAt(got, x-b.Min.X, y-b.Min.Y)
			for _, d := range []float64{wr - gr, wg - gg, wb - gb} {
				sum += d * d
			}
		}
	}
	mse := sum / float64(3*b.Dx()*b.Dy())
	if mse == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(255*255/mse)
}

func rgbAt(m image.Image, x, y int) (r, g, b float64) {
	switch m := m.(type) {
	case *image.YCbCr:
		c := m.YCbCrAt(x, y)
		return limitedToRGB(c.Y, c.Cb, c.Cr)
	case *image.NYCbCrA:
		c := m.YCbCrAt(x, y)
		return limitedToRGB(c.Y, c.Cb, c.Cr)
	}
	c := color.NRGBAModel.Convert(m.At(x, y)).(color.NRGBA)
	return float64(c.R), float64(c.G), float64(c.B)
}

func limitedToRGB(y, cb, cr uint8) (r, g, b float64) {
	yy := 1.164 * (float64(y) - 16)
	u, v := float64(cb)-128, float64(cr)-128
	clamp := func(f float64) float64 { return math.Round(min(max(f, 0), 255)) }
	return clamp(yy + 1.596*v), clamp(yy - 0.391*u - 0.813*v), clamp(yy + 2.018*u)
}

func TestLossy(t *testing.T) {
	tests := []struct {
		name    string
		m       image.Image
		q       float32
		minPSNR float64
	}{
		{"photo q20", photo(300, 200, 4), 20, 30},
		{"photo q75", photo(300, 200, 4), 75, 33},
		{"photo q100", photo(300, 200, 4), 100, 34},
		{"gradient q75", gradient(256, 256), 75, 40},
		{"gradient q100", gradient(256, 256), 100, 45},
		{"odd size", photo(37, 23, 5), 90, 24},
		{"1x1", gradient(1, 1), 75, 40},
		{"gray", grayRamp(70, 70), 90, 40},
	}
	for _, tt := range tests {
		var buf bytes.Buffer
		if err := Encode(&buf, tt.m, &Options{Quality: tt.q}); err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		got := decode(t, buf.Bytes())
		if _, ok := got.(*image.YCbCr); !ok {
			t.Errorf("%s: decoded %T, want *image.YCbCr", tt.name, got)
		}
		if p := psnr(tt.m, got); p < tt.minPSNR {
			t.Errorf("%s: PSNR %.2f dB, want at least %v", tt.name, p, tt.minPSNR)
		}
	}
}

func grayRamp(w, h int) *image.Gray {
	m := image.NewGray(image.Rect(0, 0, w, h))
	for i := range m.Pix {
		m.Pix[i] = uint8(i % w * 255 / w)
	}
	return m
}

func TestLossyQualityOrdersSize(t *testing.T) {
	m := photo(128, 128, 6)
	prev := 0
	for _, q := range []float32{10, 40, 70, 95} {
		var buf bytes.Buffer
		if err := Encode(&buf, m, &Options{Quality: q}); err != nil {
			t.Fatal(err)
		}
		if buf.Len() <= prev {
			t.Errorf("q%v: %d bytes, not larger than %d at lower quality", q, buf.Len(), prev)
		}
		prev = buf.Len()
	}
}

func TestLossyAlpha(t *testing.T) {
	m := transparent(90, 70)
	var buf bytes.Buffer
	if err := Encode(&buf, m, &Options{Quality: 80}); err != nil {
		t.Fatal(err)
	}
	got, ok := decode(t, buf.Bytes()).(*image.NYCbCrA)
	if !ok {
		t.Fatalf("decoded %T, want *image.NYCbCrA", got)
	}
	for y := 0; y < 70; y++ {
		for x := 0; x < 90; x++ {
			if a, want := got.A[got.AOffset(x, y)], m.NRGBAAt(x, y).A; a != want {
				t.Fatalf("alpha at (%d, %d) = %d, want %d", x, y, a, want)
			}
		}
	}
}

// TestLossyReconstruction checks that the encoder's reconstruction matches
// the decoder's output bit for bit when the loop filter is off.
func TestLossyReconstruction(t *testing.T) {
	for _, m := range []*image.NRGBA{photo(67, 45, 1), noise(40, 33, 2), gradient(16, 16), gradient(1, 1), photo(200, 130, 3)} {
		w, h := m.Bounds().Dx(), m.Bounds().Dy()
	cases:
		for i, q := range []float32{1, 30, 75, 100} {
			e := newVP8Encoder(toARGB(m), w, h, q)
			e.filterLevel = 0
			e.noSubblocks = i == 2
			frame, err := e.encode()
			if err != nil {
				t.Fatal(err)
			}

			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(frame), len(frame))
			if _, err := d.DecodeFrameHeader(); err != nil {
				t.Fatal(err)
			}
			got, err := d.DecodeFrame()
			if err != nil {
				t.Fatal(err)
			}
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if g, want := got.Y[y*got.YStride+x], e.ry[y*e.ystride+x]; g != want {
						t.Errorf("%dx%d q%v: Y(%d, %d) = %d, want %d", w, h, q, x, y, g, want)
						continue cases
					}
				}
			}
			for y := 0; y < (h+1)/2; y++ {
				for x := 0; x < (w+1)/2; x++ {
					gi, ei := y*got.CStride+x, y*e.cstride+x
					if got.Cb[gi] != e.ru[ei] || got.Cr[gi] != e.rv[ei] {
						t.Errorf("%dx%d q%v: chroma (%d, %d) differs", w, h, q, x, y)
						continue cases
					}
				}
			}
		}
	}
}
