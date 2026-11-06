// Package webp implements a WebP encoder in pure Go.
//
// Lossless images use the VP8L format and decode to exactly the pixels that
// were encoded. Lossy images use VP8 with an optional lossless alpha plane.
// Use golang.org/x/image/webp to decode.
package webp

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
)

// MaxSize is the largest width or height WebP can store.
const MaxSize = 16383

// DefaultQuality is used when Options.Quality is zero.
const DefaultQuality = 75

// Options are the encoding parameters.
type Options struct {
	// Lossless selects the VP8L format.
	Lossless bool

	// Quality ranges from 0 to 100. For lossy images it trades size for
	// fidelity; for lossless images it trades speed for size. Zero means
	// DefaultQuality, so the lowest usable setting is just above it.
	Quality float32

	// Exact keeps the RGB values of fully transparent pixels. Without it
	// they may be changed to compress better.
	Exact bool
}

// ErrInvalidSize is returned for images that are empty or larger than
// MaxSize in either dimension.
var ErrInvalidSize = errors.New("webp: invalid image size")

// Encode writes m to w in WebP format. A nil Options encodes losslessly with
// default settings.
func Encode(w io.Writer, m image.Image, o *Options) error {
	if o == nil {
		o = &Options{Lossless: true}
	}
	b := m.Bounds()
	if b.Dx() < 1 || b.Dy() < 1 || b.Dx() > MaxSize || b.Dy() > MaxSize {
		return fmt.Errorf("%w: %dx%d", ErrInvalidSize, b.Dx(), b.Dy())
	}
	q := o.Quality
	if q == 0 {
		q = DefaultQuality
	}
	if q < 0 || q > 100 {
		return fmt.Errorf("webp: quality %v out of range [0, 100]", o.Quality)
	}

	argb := toARGB(m)
	if o.Lossless {
		if !o.Exact {
			for i, p := range argb {
				if p>>24 == 0 {
					argb[i] = 0
				}
			}
		}
		data := encodeLossless(argb, b.Dx(), b.Dy(), q)
		return writeRIFF(w, chunk{"VP8L", data})
	}
	if !o.Exact {
		smoothTransparent(argb, b.Dx(), b.Dy())
	}
	return encodeLossy(w, argb, b.Dx(), b.Dy(), q)
}

type chunk struct {
	id   string
	data []byte
}

func writeRIFF(w io.Writer, chunks ...chunk) error {
	size := 4
	for _, c := range chunks {
		size += 8 + len(c.data) + len(c.data)&1
	}
	if int64(size) > 1<<32-2 {
		return errors.New("webp: image too large for RIFF container")
	}

	hdr := make([]byte, 12, 20)
	copy(hdr, "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:], uint32(size))
	copy(hdr[8:], "WEBP")
	for _, c := range chunks {
		hdr = append(hdr, c.id...)
		hdr = binary.LittleEndian.AppendUint32(hdr, uint32(len(c.data)))
		if _, err := w.Write(hdr); err != nil {
			return err
		}
		if _, err := w.Write(c.data); err != nil {
			return err
		}
		hdr = hdr[:0]
		if len(c.data)&1 == 1 {
			hdr = append(hdr, 0)
		}
	}
	if len(hdr) > 0 {
		_, err := w.Write(hdr)
		return err
	}
	return nil
}

// toARGB returns the non-premultiplied pixels of m packed as 0xAARRGGBB.
func toARGB(m image.Image) []uint32 {
	b := m.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint32, 0, w*h)

	switch m := m.(type) {
	case *image.NRGBA:
		for y := b.Min.Y; y < b.Max.Y; y++ {
			row := m.Pix[m.PixOffset(b.Min.X, y):][:4*w]
			for i := 0; i < len(row); i += 4 {
				out = append(out, uint32(row[i+3])<<24|uint32(row[i])<<16|uint32(row[i+1])<<8|uint32(row[i+2]))
			}
		}
	case *image.RGBA:
		for y := b.Min.Y; y < b.Max.Y; y++ {
			row := m.Pix[m.PixOffset(b.Min.X, y):][:4*w]
			for i := 0; i < len(row); i += 4 {
				out = append(out, unpremultiply(row[i], row[i+1], row[i+2], row[i+3]))
			}
		}
	default:
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				c := color.NRGBAModel.Convert(m.At(x, y)).(color.NRGBA)
				out = append(out, uint32(c.A)<<24|uint32(c.R)<<16|uint32(c.G)<<8|uint32(c.B))
			}
		}
	}
	return out
}

func unpremultiply(r, g, b, a uint8) uint32 {
	switch a {
	case 0:
		return 0
	case 0xff:
		return 0xff000000 | uint32(r)<<16 | uint32(g)<<8 | uint32(b)
	}
	c := color.NRGBAModel.Convert(color.RGBA{r, g, b, a}).(color.NRGBA)
	return uint32(a)<<24 | uint32(c.R)<<16 | uint32(c.G)<<8 | uint32(c.B)
}
