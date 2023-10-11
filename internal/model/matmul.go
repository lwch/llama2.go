package model

import (
	"sync"
)

func matMul(x, w []float32, m, n, d int64, output []float32) {
	if m > n {
		rowParallelMatMul(x, w, m, n, d, output)
	} else {
		colParallelMatMul(x, w, m, n, d, output)
	}
}

func rowParallelMatMul(x, w []float32, m, n, d int64, output []float32) {
	var wg sync.WaitGroup
	wg.Add(int(m))
	for row := int64(0); row < m; row++ {
		go func(row int64) {
			defer wg.Done()
			for col := int64(0); col < n; col++ {
				idx := row*n + col
				output[idx] = 0
				for i := int64(0); i < d; i++ {
					output[idx] += x[row*d+i] * w[i*n+col]
				}
			}
		}(row)
	}
	wg.Wait()
}

func colParallelMatMul(x, w []float32, m, n, d int64, output []float32) {
	var wg sync.WaitGroup
	wg.Add(int(n))
	for col := int64(0); col < n; col++ {
		go func(col int64) {
			defer wg.Done()
			for row := int64(0); row < m; row++ {
				idx := row*n + col
				output[idx] = 0
				for i := int64(0); i < d; i++ {
					output[idx] += x[row*d+i] * w[i*n+col]
				}
			}
		}(col)
	}
	wg.Wait()
}
