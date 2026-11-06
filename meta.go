package webp

import (
	"math"
	"slices"
)

const (
	maxMetaTiles  = 2048
	maxMetaGroups = 32
)

// metaBits picks the tile size of the entropy image so that the number of
// tiles stays bounded.
func metaBits(w, h int) int {
	bits := 5
	for bits < 9 && tiles(w, bits)*tiles(h, bits) > maxMetaTiles {
		bits++
	}
	return bits
}

// forEachTile calls fn with the entropy image tile in which each ref starts.
func forEachTile(refs []ref, w, bits int, fn func(tile int, r ref)) {
	tw := tiles(w, bits)
	x, y := 0, 0
	for _, r := range refs {
		fn((y>>bits)*tw+x>>bits, r)
		n := 1
		if r.kind == refCopy {
			n = int(r.val)
		}
		for x += n; x >= w; x -= w {
			y++
		}
	}
}

// clusterTiles groups the tiles of the image into histograms that are cheap
// to code together. It returns one group when splitting does not pay off.
func clusterTiles(refs []ref, w, h, cacheBits int) (bits int, tileGroup []uint32, groups []histogram) {
	bits = metaBits(w, h)
	ntiles := tiles(w, bits) * tiles(h, bits)

	tileHist := make([]histogram, ntiles)
	for i := range tileHist {
		tileHist[i] = newHistogram(cacheBits)
	}
	forEachTile(refs, w, bits, func(t int, r ref) { tileHist[t].add(r) })

	var (
		used   []int
		sparse [][]int32
	)
	for i := range tileHist {
		var nz []int32
		for j, n := range tileHist[i].all {
			if n > 0 {
				nz = append(nz, int32(j))
			}
		}
		if nz != nil {
			used = append(used, i)
			sparse = append(sparse, nz)
		}
	}
	cross := func(i int, lens []float32) float64 {
		all := tileHist[used[i]].all
		var c float32
		for _, j := range sparse[i] {
			c += float32(all[j]) * lens[j]
		}
		return float64(c)
	}
	tileGroup = make([]uint32, ntiles)
	if len(used) < 2 {
		return bits, tileGroup, nil
	}

	k := min(maxMetaGroups, len(used))
	assign := make([]int, len(used))
	clusters := make([]histogram, 0, k)
	lens := make([][]float32, 0, k)

	// Seed with the busiest tile, then repeatedly with the tile that the
	// current clusters code worst.
	first := slices.MaxFunc(used, func(a, b int) int { return int(tileHist[a].total()) - int(tileHist[b].total()) })
	cost := make([]float64, len(used))
	for i := range cost {
		cost[i] = math.Inf(1)
	}
	seed := first
	for len(clusters) < k {
		c := newHistogram(cacheBits)
		c.merge(&tileHist[seed])
		clusters = append(clusters, c)
		lens = append(lens, codeLengths(&c))
		ci := len(clusters) - 1
		worst, worstCost := -1, 0.0
		for i := range used {
			if c := cross(i, lens[ci]); c < cost[i] {
				cost[i], assign[i] = c, ci
			}
			if cost[i] > worstCost {
				worst, worstCost = i, cost[i]
			}
		}
		if worst < 0 || worstCost == 0 {
			break
		}
		seed = used[worst]
	}

	rebuild := func() {
		for i := range clusters {
			clear(clusters[i].all)
			clusters[i].extraBits = 0
		}
		for i, t := range used {
			clusters[assign[i]].merge(&tileHist[t])
		}
	}
	for iter := 0; iter < 3; iter++ {
		rebuild()
		for i := range clusters {
			lens[i] = codeLengths(&clusters[i])
		}
		for i := range used {
			best, bestCost := assign[i], math.Inf(1)
			for ci := range clusters {
				if c := cross(i, lens[ci]); c < bestCost {
					best, bestCost = ci, c
				}
			}
			assign[i] = best
		}
	}
	rebuild()

	clusters, remap := mergeClusters(clusters, cacheBits)
	if len(clusters) < 2 {
		return bits, tileGroup, nil
	}
	single := newHistogram(cacheBits)
	split := float64(ntiles)*math.Log2(float64(len(clusters)))/2 + 100
	for i := range clusters {
		single.merge(&clusters[i])
		split += groupCost(&clusters[i])
	}
	if split >= groupCost(&single) {
		return bits, tileGroup, nil
	}
	for i, t := range used {
		tileGroup[t] = uint32(remap[assign[i]])
	}
	return bits, tileGroup, clusters
}

// mergeClusters greedily merges the pair of clusters that saves the most
// bits until no merge helps. remap gives the new index of each old cluster.
func mergeClusters(clusters []histogram, cacheBits int) ([]histogram, []int) {
	remap := make([]int, len(clusters))
	for i := range remap {
		remap[i] = i
	}
	alive := make([]bool, len(clusters))
	cost := make([]float64, len(clusters))
	for i := range clusters {
		alive[i] = true
		cost[i] = groupCost(&clusters[i])
	}

	n := len(clusters)
	saving := make([]float64, n*n)
	pairSaving := func(a, b int) float64 {
		m := newHistogram(cacheBits)
		m.merge(&clusters[a])
		m.merge(&clusters[b])
		return cost[a] + cost[b] - groupCost(&m)
	}
	for a := 0; a < n; a++ {
		for b := a + 1; b < n; b++ {
			saving[a*n+b] = pairSaving(a, b)
		}
	}

	for {
		ba, bb, best := -1, -1, 0.0
		for a := 0; a < n; a++ {
			for b := a + 1; b < n && alive[a]; b++ {
				if alive[b] && saving[a*n+b] > best {
					ba, bb, best = a, b, saving[a*n+b]
				}
			}
		}
		if ba < 0 {
			break
		}
		clusters[ba].merge(&clusters[bb])
		cost[ba] = groupCost(&clusters[ba])
		alive[bb] = false
		for i := range remap {
			if remap[i] == bb {
				remap[i] = ba
			}
		}
		for o := 0; o < n; o++ {
			if o != ba && alive[o] {
				s := pairSaving(min(o, ba), max(o, ba))
				saving[min(o, ba)*n+max(o, ba)] = s
			}
		}
	}

	var out []histogram
	index := make([]int, n)
	for i := range clusters {
		if alive[i] {
			index[i] = len(out)
			out = append(out, clusters[i])
		}
	}
	for i := range remap {
		remap[i] = index[remap[i]]
	}
	return out, remap
}

func (h *histogram) merge(o *histogram) {
	for i, n := range o.all {
		h.all[i] += n
	}
	h.extraBits += o.extraBits
}

func (h *histogram) total() uint32 {
	var t uint32
	for _, n := range h.green {
		t += n
	}
	return t
}

// groupCost estimates the bits for a group's data plus its code headers.
func groupCost(h *histogram) float64 {
	c := 0.0
	for _, p := range h.parts() {
		c += entropy(p) + headerCost(p)
	}
	return c
}

// headerCost roughly estimates the size of a prefix code header.
func headerCost(hist []uint32) float64 {
	nz, runs := 0, 0
	zero := false
	for _, n := range hist {
		if n > 0 {
			nz++
			zero = false
		} else if !zero {
			runs++
			zero = true
		}
	}
	if nz <= 2 {
		return 12
	}
	return 60 + 3.5*float64(nz) + 6*float64(runs)
}

// codeLengths returns smoothed code length estimates for each bin of h, with
// each prefix code normalised separately.
func codeLengths(h *histogram) []float32 {
	out := make([]float32, 0, len(h.all))
	for _, p := range h.parts() {
		var total float64
		for _, n := range p {
			total += float64(n)
		}
		total += 0.1 * float64(len(p))
		for _, n := range p {
			out = append(out, float32(math.Log2(total/(float64(n)+0.1))))
		}
	}
	return out
}
