package webp

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"

	"golang.org/x/image/webp"
)

func gradient(w, h int) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetNRGBA(x, y, color.NRGBA{uint8(x * 255 / max(w-1, 1)), uint8(y * 255 / max(h-1, 1)), uint8((x + y) / 2), 255})
		}
	}
	return m
}

func solid(w, h int, c color.NRGBA) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(m.Pix); i += 4 {
		m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = c.R, c.G, c.B, c.A
	}
	return m
}

func noise(w, h int, seed int64) *image.NRGBA {
	r := rand.New(rand.NewSource(seed))
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	r.Read(m.Pix)
	return m
}

// photo is smooth shapes with mild noise, like a downscaled photograph.
func photo(w, h int, seed int64) *image.NRGBA {
	r := rand.New(rand.NewSource(seed))
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	fw, fh := float64(w), float64(h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x)/fw, float64(y)/fh
			v := math.Sin(fx*7+fy*3) + math.Cos(fx*fy*11)
			d := math.Hypot(fx-0.6, fy-0.4)
			c := func(base, amp float64) uint8 {
				return uint8(min(max(base+amp*v+r.NormFloat64()*3, 0), 255))
			}
			if d < 0.25 {
				m.SetNRGBA(x, y, color.NRGBA{c(200, 30), c(80, 20), c(60, 20), 255})
			} else {
				m.SetNRGBA(x, y, color.NRGBA{c(90, 40), c(140, 50), c(170, 30), 255})
			}
		}
	}
	return m
}

func paletted(w, h, n int, seed int64) *image.NRGBA {
	r := rand.New(rand.NewSource(seed))
	pal := make([]color.NRGBA, n)
	for i := range pal {
		pal[i] = color.NRGBA{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256))}
	}
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (x/7 + y/5 + r.Intn(2)) % n
			m.SetNRGBA(x, y, pal[i])
		}
	}
	return m
}

func transparent(w, h int) *image.NRGBA {
	m := photo(w, h, 3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := m.PixOffset(x, y)
			dx, dy := float64(x-w/2), float64(y-h/2)
			a := 255 - min(255, int(math.Hypot(dx, dy)*512/float64(w)))
			m.Pix[i+3] = uint8(a)
		}
	}
	return m
}

func decode(t *testing.T, data []byte) image.Image {
	t.Helper()
	m, err := webp.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return m
}

// sameNRGBA reports the first pixel where got differs from want. When exact
// is false, fully transparent pixels only need to stay transparent.
func sameNRGBA(t *testing.T, want image.Image, got image.Image, exact bool) {
	t.Helper()
	b := want.Bounds()
	if got.Bounds().Dx() != b.Dx() || got.Bounds().Dy() != b.Dy() {
		t.Fatalf("size = %v, want %v", got.Bounds(), b)
	}
	g0 := got.Bounds().Min
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			w := color.NRGBAModel.Convert(want.At(x, y)).(color.NRGBA)
			g := color.NRGBAModel.Convert(got.At(g0.X+x-b.Min.X, g0.Y+y-b.Min.Y)).(color.NRGBA)
			if w != g && (exact || w.A != 0 || g.A != 0) {
				t.Fatalf("pixel (%d, %d) = %v, want %v", x, y, g, w)
			}
		}
	}
}

func TestLosslessRoundTrip(t *testing.T) {
	gray := image.NewGray(image.Rect(0, 0, 33, 17))
	for i := range gray.Pix {
		gray.Pix[i] = uint8(i * 7)
	}
	rgba := image.NewRGBA(image.Rect(0, 0, 40, 30))
	rand.New(rand.NewSource(9)).Read(rgba.Pix)
	for i := 0; i < len(rgba.Pix); i += 4 {
		a := rgba.Pix[i+3]
		rgba.Pix[i] = min(rgba.Pix[i], a)
		rgba.Pix[i+1] = min(rgba.Pix[i+1], a)
		rgba.Pix[i+2] = min(rgba.Pix[i+2], a)
	}
	sub := gradient(64, 64).SubImage(image.Rect(5, 7, 50, 41))

	tests := []struct {
		name string
		m    image.Image
	}{
		{"1x1", gradient(1, 1)},
		{"1x1 transparent", image.NewNRGBA(image.Rect(0, 0, 1, 1))},
		{"2x1", noise(2, 1, 1)},
		{"1x9", noise(1, 9, 2)},
		{"gradient", gradient(256, 256)},
		{"gradient odd", gradient(257, 131)},
		{"noise", noise(127, 93, 3)},
		{"photo", photo(300, 200, 4)},
		{"palette 2", paletted(100, 61, 2, 5)},
		{"palette 3", paletted(99, 50, 3, 6)},
		{"palette 16", paletted(101, 77, 16, 7)},
		{"palette 17", paletted(64, 64, 17, 8)},
		{"palette 256", paletted(129, 129, 256, 9)},
		{"solid", solid(70, 70, color.NRGBA{10, 20, 30, 40})},
		{"transparent", transparent(150, 120)},
		{"gray", gray},
		{"rgba", rgba},
		{"subimage", sub},
		{"wide", noise(5000, 3, 10)},
		{"tall", gradient(2, 3000)},
	}
	for _, tt := range tests {
		for _, q := range []float32{1, 75, 100} {
			m := tt.m
			var buf bytes.Buffer
			if err := Encode(&buf, m, &Options{Lossless: true, Quality: q, Exact: true}); err != nil {
				t.Fatalf("%s q%v: %v", tt.name, q, err)
			}
			sameNRGBA(t, m, decode(t, buf.Bytes()), true)
		}
	}
}

func TestLosslessRandom(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 60; i++ {
		w, h := 1+r.Intn(70), 1+r.Intn(70)
		var m *image.NRGBA
		switch i % 4 {
		case 0:
			m = noise(w, h, int64(i))
		case 1:
			m = paletted(w, h, 1+r.Intn(300), int64(i))
		case 2:
			m = photo(w, h, int64(i))
		default:
			m = transparent(w, h)
		}
		q := float32(r.Intn(101))
		var buf bytes.Buffer
		if err := Encode(&buf, m, &Options{Lossless: true, Quality: q, Exact: true}); err != nil {
			t.Fatal(err)
		}
		sameNRGBA(t, m, decode(t, buf.Bytes()), true)

		buf.Reset()
		if err := Encode(&buf, m, &Options{Quality: q}); err != nil {
			t.Fatal(err)
		}
		decode(t, buf.Bytes())
	}
}

func TestLosslessNotExact(t *testing.T) {
	m := noise(64, 64, 11)
	for i := 3; i < len(m.Pix); i += 8 {
		m.Pix[i] = 0
	}
	var buf bytes.Buffer
	if err := Encode(&buf, m, &Options{Lossless: true}); err != nil {
		t.Fatal(err)
	}
	sameNRGBA(t, m, decode(t, buf.Bytes()), false)
}

func TestInvalidSize(t *testing.T) {
	for _, r := range []image.Rectangle{
		image.Rect(0, 0, 0, 10),
		image.Rect(0, 0, 10, 0),
		image.Rect(0, 0, MaxSize+1, 1),
		image.Rect(0, 0, 1, MaxSize+1),
	} {
		err := Encode(&bytes.Buffer{}, image.NewNRGBA(r), nil)
		if !errors.Is(err, ErrInvalidSize) {
			t.Errorf("%v: err = %v, want ErrInvalidSize", r, err)
		}
	}
}

func TestMaxSize(t *testing.T) {
	if testing.Short() {
		t.Skip("large image")
	}
	m := gradient(MaxSize, 2)
	var buf bytes.Buffer
	if err := Encode(&buf, m, nil); err != nil {
		t.Fatal(err)
	}
	sameNRGBA(t, m, decode(t, buf.Bytes()), true)
}

func TestQualityRange(t *testing.T) {
	for _, q := range []float32{-1, 101} {
		if err := Encode(&bytes.Buffer{}, gradient(4, 4), &Options{Lossless: true, Quality: q}); err == nil {
			t.Errorf("quality %v: no error", q)
		}
	}
}

func benchmarkEncode(b *testing.B, m image.Image, o *Options) {
	var buf bytes.Buffer
	b.ReportAllocs()
	b.SetBytes(int64(4 * m.Bounds().Dx() * m.Bounds().Dy()))
	for i := 0; i < b.N; i++ {
		buf.Reset()
		if err := Encode(&buf, m, o); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(buf.Len()), "bytes")
}

func BenchmarkEncodeLossless(b *testing.B) {
	benchmarkEncode(b, photo(1024, 1024, 1), &Options{Lossless: true})
}

func BenchmarkEncodeLosslessPalette(b *testing.B) {
	benchmarkEncode(b, paletted(1024, 1024, 64, 1), &Options{Lossless: true})
}

func BenchmarkEncodeLossy(b *testing.B) {
	benchmarkEncode(b, photo(1024, 1024, 1), &Options{Quality: 75})
}

func BenchmarkEncodeLossyAlpha(b *testing.B) {
	benchmarkEncode(b, transparent(1024, 1024), &Options{Quality: 75})
}
