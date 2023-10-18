package math

import (
	"math"
	"runtime"
	"sync"
)

// MatMul (m, d) @ (d, n) => (m, n)
func MatMul(x, w []float32, m, n, d int64, output []float32) {
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

func rmsnormScale(x []float32) float32 {
	n := runtime.NumCPU()
	step := len(x) / n
	var wg sync.WaitGroup
	var scale float32
	var m sync.Mutex
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			var sum float32
			for i := 0; i < step; i++ {
				n := float64(x[offset+i])
				sum += float32(math.Pow(n, 2))
			}
			m.Lock()
			scale += sum
			m.Unlock()
		}(offset, step)
	}
	wg.Wait()
	scale /= float32(len(x))
	return scale
}

func RMSNorm(x, w []float32, output []float32, eps float32) {
	scale := rmsnormScale(x)
	scale += eps
	scale = 1 / float32(math.Sqrt(float64(scale)))
	n := runtime.NumCPU()
	step := len(x) / n
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			for i := 0; i < step; i++ {
				idx := offset + i
				output[idx] = x[idx] * scale * w[idx]
			}
		}(offset, step)
	}
	wg.Wait()
}

func softmaxMax(x []float32) float32 {
	n := runtime.NumCPU()
	step := len(x) / n
	var wg sync.WaitGroup
	wg.Add(n)
	max := x[0]
	var m sync.Mutex
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			for i := 0; i < step; i++ {
				idx := offset + i
				m.Lock()
				if x[idx] > max {
					max = x[idx]
				}
				m.Unlock()
			}
		}(offset, step)
	}
	wg.Wait()
	return max
}

func softmaxSum(x []float32, max float32) float32 {
	n := runtime.NumCPU()
	step := len(x) / n
	var wg sync.WaitGroup
	wg.Add(n)
	var sum float32
	var m sync.Mutex
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			for i := 0; i < step; i++ {
				idx := offset + i
				dx := float32(math.Exp(float64(x[idx] - max)))
				x[idx] = dx
				m.Lock()
				sum += dx
				m.Unlock()
			}
		}(offset, step)
	}
	wg.Wait()
	return sum
}

func softmax(x []float32, sum float32) {
	n := runtime.NumCPU()
	step := len(x) / n
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			for i := 0; i < step; i++ {
				idx := offset + i
				x[idx] /= sum
			}
		}(offset, step)
	}
	wg.Wait()
}

func Softmax(x []float32, n int64) {
	max := softmaxMax(x)
	sum := softmaxSum(x, max)
	softmax(x, sum)
}

func SiLU(x []float32) {
	n := runtime.NumCPU()
	step := len(x) / n
	silu := func(offset, size int) {
		for i := 0; i < size; i++ {
			idx := offset + i
			x[idx] *= Sigmoid(x[idx])
		}
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			silu(offset, step)
		}(offset, step)
	}
	wg.Wait()
}

func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

// Mul x * w => x
func Mul(x, w []float32) {
	n := runtime.NumCPU()
	step := len(x) / n
	mul := func(offset, size int) {
		for i := 0; i < size; i++ {
			idx := offset + i
			x[idx] *= w[idx]
		}
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			mul(offset, step)
		}(offset, step)
	}
	wg.Wait()
}

// ROPE code from https://github.com/karpathy/llama2.c/blob/master/run.c#L265
func ROPE(q, k []float32, cursor, headSize int64) {
	n := runtime.NumCPU()
	step := len(q) / n / 2
	rope := func(offset, size int) {
		for i := 0; i < size; i += 2 {
			headDim := int64(offset+i) % headSize
			freq := 1 / math.Pow(10000, float64(headDim)/float64(headSize))
			val := float64(cursor) * freq
			fcr := float32(math.Cos(val))
			fci := float32(math.Sin(val))
			set := func(x []float32) {
				idx := offset + i
				if idx >= len(x) {
					return
				}
				v0 := x[i]
				v1 := x[i+1]
				x[i] = float32(fcr*v0 - fci*v1)
				x[i+1] = float32(fcr*v1 + fci*v0)
			}
			set(q)
			set(k)
		}
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step * 2
		if i == n-1 {
			step = (len(q) - offset) / 2
		}
		go func(offset, step int) {
			defer wg.Done()
			rope(offset, step)
		}(offset, step*2)
	}
	wg.Wait()
}

// Add x + w => x
func Add(x, w []float32) {
	n := runtime.NumCPU()
	step := len(x) / n
	add := func(offset, size int) {
		for i := 0; i < size; i++ {
			idx := offset + i
			x[idx] += w[idx]
		}
	}
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		offset := i * step
		if i == n-1 {
			step = len(x) - offset
		}
		go func(offset, step int) {
			defer wg.Done()
			add(offset, step)
		}(offset, step)
	}
}
