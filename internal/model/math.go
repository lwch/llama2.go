package model

import (
	"math"
	"sync"
)

// matMul (m, d) @ (d, n) => (m, n)
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

func rmsnorm(x, w []float32, output []float32, eps float32) {
	var scale float32
	for i := 0; i < len(x); i++ {
		scale += x[i] * x[i]
	}
	scale /= float32(len(x))
	scale += eps
	scale = 1 / float32(math.Sqrt(float64(scale)))
	var wg sync.WaitGroup
	wg.Add(len(x))
	for i, v := range x {
		go func(i int, v float32) {
			defer wg.Done()
			output[i] = v * scale * w[i]
		}(i, v)
	}
	wg.Wait()
}

func softmax(x []float32, n int64) {
	max := x[0]
	for i := int64(0); i < n; i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	var sum float32
	for i := int64(0); i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	for i := int64(0); i < n; i++ {
		x[i] /= sum
	}
}

func silu(x []float32) {
	var wg sync.WaitGroup
	wg.Add(len(x))
	for i := range x {
		go func(i int) {
			defer wg.Done()
			x[i] = x[i] * sigmoid(x[i])
		}(i)
	}
	wg.Wait()
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func mul(x, w []float32, output []float32) {
	var wg sync.WaitGroup
	wg.Add(len(x))
	for i, v := range x {
		go func(i int, v float32) {
			defer wg.Done()
			output[i] = v * w[i]
		}(i, v)
	}
	wg.Wait()
}
