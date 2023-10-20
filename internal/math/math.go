package math

import (
	"llama2/internal/utils"
	"runtime"

	"github.com/chewxy/math32"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// MatMul (m, d) @ (d, n) => (m, n)
func MatMul(x, w []float32, m, n, d int64, output []float32) {
	bx := blas32.General{
		Rows:   int(m),
		Cols:   int(d),
		Stride: int(d),
		Data:   x,
	}
	bw := blas32.General{
		Rows:   int(n),
		Cols:   int(d),
		Stride: int(d),
		Data:   w,
	}
	bout := blas32.General{
		Rows:   int(m),
		Cols:   int(n),
		Stride: int(n),
		Data:   output,
	}
	blas32.Gemm(blas.NoTrans, blas.Trans, 1, bx, bw, 0, bout)
}

// AXPY y = a * x + y
func Axpy(a float32, x, y []float32) {
	bx := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}
	by := blas32.Vector{
		N:    len(y),
		Inc:  1,
		Data: y,
	}
	blas32.Axpy(a, bx, by)
}

func RMSNorm(x, w []float32, output []float32, eps float32) {
	bx := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}
	scale := blas32.Dot(bx, bx)
	scale /= float32(len(x))
	scale += eps
	scale = 1 / math32.Sqrt(scale)
	utils.Parallel(len(x), runtime.NumCPU(), func(_, offset, size int) {
		end := offset + size
		for i := offset; i < end; i++ {
			output[i] = scale * x[i] * w[i]
		}
	})
}

func Softmax(x []float32, n int64) {
	// get max
	values := make([]float32, runtime.NumCPU())
	utils.Parallel(int(n), runtime.NumCPU(), func(batch, offset, size int) {
		max := x[offset]
		end := offset + size
		for i := offset + 1; i < end; i++ {
			if x[i] > max {
				max = x[i]
			}
		}
		values[batch] = max
	})
	max := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > max {
			max = values[i]
		}
	}
	// exp
	Clear(values)
	utils.Parallel(int(n), runtime.NumCPU(), func(batch, offset, size int) {
		var sum float32
		end := offset + size
		for i := offset; i < end; i++ {
			dx := math32.Exp(x[i] - max)
			x[i] = dx
			sum += dx
		}
		values[batch] = sum
	})
	sum := values[0]
	for i := 1; i < len(values); i++ {
		sum += values[i]
	}
	// normalize
	bx := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}
	blas32.Scal(1/sum, bx)
}

func SiLU(x []float32) {
	utils.Parallel(len(x), runtime.NumCPU(), func(_, offset, size int) {
		end := offset + size
		for i := offset; i < end; i++ {
			x[i] *= Sigmoid(x[i])
		}
	})
}

func Sigmoid(x float32) float32 {
	return 1 / (1 + math32.Exp(-x))
}

// Mul x * w => x
func Mul(x, w []float32) {
	utils.Parallel(len(x), runtime.NumCPU(), func(_, offset, size int) {
		end := offset + size
		for i := offset; i < end; i++ {
			x[i] *= w[i]
		}
	})
}

// ROPE code from https://github.com/karpathy/llama2.c/blob/master/run.c#L265
func ROPE(q, k []float32, cursor, headSize int64) {
	set := func(x []float32, i int, fcr, fci float32) {
		if i >= len(x) {
			return
		}
		v0 := x[i]
		v1 := x[i+1]
		x[i] = fcr*v0 - fci*v1
		x[i+1] = fcr*v1 + fci*v0
	}
	utils.Parallel(len(q)/2, runtime.NumCPU(), func(_, offset, size int) {
		for i := 0; i < size; i++ {
			idx := (offset + i) * 2
			headDim := int64(idx) % headSize
			freq := 1 / math32.Pow(10000, float32(headDim)/float32(headSize))
			val := float32(cursor) * freq
			fcr := math32.Cos(val)
			fci := math32.Sin(val)
			set(q, idx, fcr, fci)
			set(k, idx, fcr, fci)
		}
	})
}

// Add x + w => x
func Add(x, w []float32) {
	bx := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}
	bw := blas32.Vector{
		N:    len(w),
		Inc:  1,
		Data: w,
	}
	blas32.Axpy(1, bw, bx)
}

func Clear(x []float32) {
	bx := blas32.Vector{
		N:    len(x),
		Inc:  1,
		Data: x,
	}
	blas32.Scal(0, bx)
}
