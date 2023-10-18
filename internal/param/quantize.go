package param

import (
	"sync"
)

func Quantize(values []float32, t Type, groupSize int64) ([]byte, []byte) {
	switch t {
	case TypeQI8:
		return quantizeInt8(values, groupSize)
	}
	return nil, nil
}

func parallel(size, batches int, fn func(batch, offset, size int)) {
	step := size / batches
	var wg sync.WaitGroup
	wg.Add(batches)
	for i := 0; i < batches; i++ {
		offset := i * step
		if i == batches-1 {
			step = size - offset
		}
		go func(i, offset, step int) {
			defer wg.Done()
			fn(i, offset, step)
		}(i, offset, step)
	}
	wg.Wait()
}
