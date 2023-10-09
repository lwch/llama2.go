package parallel

import (
	"sync"

	"github.com/lwch/gotorch/tensor"
)

func MatMul(a, b *tensor.Tensor) *tensor.Tensor {
	rows := a.Shapes()[a.Dims()-2]
	data := make([]*tensor.Tensor, rows)
	var wg sync.WaitGroup
	wg.Add(int(rows))
	for i := int64(0); i < rows; i++ {
		go func(i int64) {
			defer wg.Done()
			data[i] = a.NArrow(-2, i, 1).MatMul(b)
		}(i)
	}
	wg.Wait()
	return tensor.Cat(data, -2)
}
