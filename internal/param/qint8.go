package param

import (
	"bytes"
	"encoding/binary"
	"llama2/internal/utils"
	"math"
	rt "runtime"

	"github.com/lwch/runtime"
)

func quantizeInt8(values []float32, groupSize int64) ([]byte, []byte) {
	scale := make([]float32, len(values)/int(groupSize))
	for i := range scale {
		scale[i] = values[i*int(groupSize)]
	}
	utils.Parallel(len(values), rt.NumCPU(), func(_, offset, size int) {
		for i := 0; i < size; i++ {
			idx := offset + i
			target := idx / int(groupSize)
			if values[idx] > scale[target] {
				scale[target] = float32(math.Abs(float64(values[idx])))
			}
		}
	})
	for i := range scale {
		scale[i] /= 127
	}
	scaled := make([]int8, len(values))
	utils.Parallel(len(values), rt.NumCPU(), func(_, offset, size int) {
		for i := 0; i < size; i++ {
			idx := offset + i
			target := idx / int(groupSize)
			scaled[idx] = int8(values[idx] / scale[target])
		}
	})
	return encodeInt(scaled), encodeScale(scale)
}

func encodeInt(values []int8) []byte {
	var buf bytes.Buffer
	runtime.Assert(binary.Write(&buf, binary.LittleEndian, values))
	return buf.Bytes()
}

func encodeScale(values []float32) []byte {
	var buf bytes.Buffer
	runtime.Assert(binary.Write(&buf, binary.LittleEndian, values))
	return buf.Bytes()
}
