# webp

A WebP encoder in pure Go. No cgo, no dependencies outside the standard library. Pairs with [`golang.org/x/image/webp`](https://pkg.go.dev/golang.org/x/image/webp), which only decodes.

```sh
go get github.com/hostfog/webp
```

## Usage

```go
f, _ := os.Create("out.webp")
defer f.Close()

err := webp.Encode(f, img, &webp.Options{Lossless: true}) // pixel-exact
err = webp.Encode(f, img, &webp.Options{Quality: 80})     // lossy
```

The API mirrors `image/png` and `image/jpeg`. A nil `*Options` encodes losslessly.

| Option     | Meaning                                                                                      |
|------------|----------------------------------------------------------------------------------------------|
| `Lossless` | VP8L instead of VP8.                                                                         |
| `Quality`  | 0–100, default 75. Lossy: size versus fidelity. Lossless: speed versus size.                 |
| `Exact`    | Keep the RGB of fully transparent pixels. Off by default, which lets them compress better.   |

## Formats

- **Lossless (VP8L):** subtract-green, predictor and cross-colour transforms, colour-indexing with pixel bundling for images of up to 256 colours, LZ77 with hash chains, colour cache, length-limited canonical Huffman codes and meta Huffman codes. Output decodes to the exact input pixels.
- **Lossy (VP8 key frame):** BT.601 YCbCr 4:2:0, rate-distortion choice of 16×16 and 4×4 intra predictors, token probability updates and the normal loop filter. Alpha is stored losslessly in an `ALPH` chunk.

Images up to 16383×16383 are supported.

## Size

512×512 generated test images (from the tests), default settings:

| Image       | PNG       | WebP lossless    | WebP lossy q75   |
|-------------|-----------|------------------|------------------|
| Gradient    | 2.2 KB    | 1.0 KB (43%)     | 3.4 KB (154%)    |
| Photo-like  | 412.8 KB  | 385.7 KB (93%)   | 4.1 KB (1%)      |
| 16 colours  | 135.9 KB  | 64.1 KB (47%)    | 197.1 KB (145%)  |
| Transparent | 516.0 KB  | 337.3 KB (65%)   | 22.6 KB (4%)     |
| Noise       | 1025.3 KB | 1024.2 KB (100%) | 418.7 KB (41%)   |

Lossless output is typically within 3–8% of libwebp at its default effort; lossy output is comparable to libwebp in size and PSNR.

## License

MIT
