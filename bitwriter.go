package webp

// bitWriter packs bits least-significant first, as VP8L requires.
type bitWriter struct {
	buf  []byte
	acc  uint64
	nacc uint
}

func (w *bitWriter) write(v uint32, n uint) {
	w.acc |= uint64(v) << w.nacc
	w.nacc += n
	for w.nacc >= 8 {
		w.buf = append(w.buf, byte(w.acc))
		w.acc >>= 8
		w.nacc -= 8
	}
}

func (w *bitWriter) bytes() []byte {
	if w.nacc > 0 {
		w.buf = append(w.buf, byte(w.acc))
		w.acc, w.nacc = 0, 0
	}
	return w.buf
}
