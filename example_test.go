package webp_test

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/hostfog/webp"
	xwebp "golang.org/x/image/webp"
)

func Example() {
	m := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			m.Set(x, y, color.NRGBA{uint8(4 * x), uint8(4 * y), 128, 255})
		}
	}

	var buf bytes.Buffer
	if err := webp.Encode(&buf, m, &webp.Options{Lossless: true}); err != nil {
		log.Fatal(err)
	}

	decoded, err := xwebp.Decode(&buf)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(decoded.At(10, 20))
	// Output: {40 80 128 255}
}

func ExampleEncode_lossy() {
	m := image.NewRGBA(image.Rect(0, 0, 320, 240))
	for i := range m.Pix {
		m.Pix[i] = 200
	}

	var buf bytes.Buffer
	if err := webp.Encode(&buf, m, &webp.Options{Quality: 80}); err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(buf.Bytes()[8:12]), buf.Len() < 1000)
	// Output: WEBP true
}
