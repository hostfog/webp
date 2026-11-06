package webp

// boolEncoder is the VP8 boolean entropy encoder (RFC 6386, section 7).
type boolEncoder struct {
	buf      []byte
	rng      uint32
	bottom   uint32
	bitCount int
}

func newBoolEncoder() *boolEncoder {
	return &boolEncoder{rng: 255, bitCount: 24}
}

func (e *boolEncoder) carry() {
	for i := len(e.buf) - 1; i >= 0; i-- {
		if e.buf[i] != 0xff {
			e.buf[i]++
			return
		}
		e.buf[i] = 0
	}
}

// putBit writes b, where prob/256 is the probability of b being false.
func (e *boolEncoder) putBit(b bool, prob uint8) {
	split := 1 + (e.rng-1)*uint32(prob)>>8
	if b {
		e.bottom += split
		e.rng -= split
	} else {
		e.rng = split
	}
	for e.rng < 128 {
		e.rng <<= 1
		if e.bottom&(1<<31) != 0 {
			e.carry()
		}
		e.bottom <<= 1
		if e.bitCount--; e.bitCount == 0 {
			e.buf = append(e.buf, byte(e.bottom>>24))
			e.bottom &= 1<<24 - 1
			e.bitCount = 8
		}
	}
}

// putLiteral writes the n low bits of v, most significant first.
func (e *boolEncoder) putLiteral(v uint32, n int) {
	for n > 0 {
		n--
		e.putBit(v>>n&1 == 1, 128)
	}
}

// finish flushes the pending bits, padding with zeros so that a decoder
// can read ahead, and returns the output.
func (e *boolEncoder) finish() []byte {
	for i := 0; i < 32; i++ {
		e.putBit(false, 128)
	}
	return e.buf
}
