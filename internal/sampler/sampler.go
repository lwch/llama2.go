package sampler

import (
	"llama2/internal/math"
	"math/rand"
	"sort"
)

type Sampler struct {
	temperature float32
	topP        float32
}

func New(temperature, topP float32) *Sampler {
	return &Sampler{
		temperature: temperature,
		topP:        topP,
	}
}

func (s *Sampler) Sample(scores []float32) uint64 {
	if s.temperature > 0 {
		for i := range scores {
			scores[i] /= s.temperature
		}
		math.Softmax(scores, int64(len(scores)))
		return s.sampleTopP(scores)
	}
	return s.argmax(scores)
}

func (s *Sampler) argmax(scores []float32) uint64 {
	max := scores[0]
	var idx uint64
	for i, score := range scores {
		if score > max {
			max = score
			idx = uint64(i)
		}
	}
	return idx
}

func (s *Sampler) sampleTopP(scores []float32) uint64 {
	cutoff := (1 - s.topP) / float32(len(scores)-1)
	type pair struct {
		idx  int
		prob float32
	}
	var index []pair
	for i := 0; i < len(scores); i++ {
		if scores[i] >= cutoff {
			index = append(index, pair{i, scores[i]})
		}
	}
	sort.Slice(index, func(i, j int) bool {
		return index[i].prob > index[j].prob
	})

	var cumulativeProb float32
	lastIdx := len(index) - 1
	for i, idx := range index {
		cumulativeProb += idx.prob
		if cumulativeProb > s.topP {
			lastIdx = i
			break
		}
	}

	r := rand.Float32() * cumulativeProb
	var cdf float32
	for i := 0; i <= lastIdx; i++ {
		cdf += index[i].prob
		if r < cdf {
			return uint64(index[i].idx)
		}
	}
	return uint64(index[lastIdx].idx)
}
