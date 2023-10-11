package model

import (
	"errors"
	"sync"
)

func matMul(x1, x2 []float32, x1Shape, x2Shape []int64, output []float32) {
	if x1Shape[len(x1Shape)-1] != x2Shape[0] {
		panic(errors.New("invalid shape"))
	}
	if x1Shape[0] > x2Shape[0] {
		rowParallelMatMul(x1, x2, x1Shape, x2Shape, output)
	} else {
		colParallelMatMul(x1, x2, x1Shape, x2Shape, output)
	}
}

func rowParallelMatMul(x1, x2 []float32, x1Shape, x2Shape []int64, output []float32) {
	x1Rows, n := x1Shape[0], x1Shape[1]
	x2Cols := x2Shape[1]
	var wg sync.WaitGroup
	wg.Add(int(x1Rows))
	for row := int64(0); row < x1Rows; row++ {
		go func(row int64) {
			defer wg.Done()
			for col := int64(0); col < x2Cols; col++ {
				idx := row*x2Cols + col
				output[idx] = 0
				for i := int64(0); i < n; i++ {
					output[idx] += x1[row*n+i] * x2[i*x2Cols+col]
				}
			}
		}(row)
	}
	wg.Wait()
}

func colParallelMatMul(x1, x2 []float32, x1Shape, x2Shape []int64, output []float32) {
	x1Rows, n := x1Shape[0], x1Shape[1]
	x2Cols := x2Shape[1]
	var wg sync.WaitGroup
	wg.Add(int(x2Cols))
	for col := int64(0); col < x2Cols; col++ {
		go func(col int64) {
			defer wg.Done()
			for row := int64(0); row < x1Rows; row++ {
				idx := row*x2Cols + col
				output[idx] = 0
				for i := int64(0); i < n; i++ {
					output[idx] += x1[row*n+i] * x2[i*x2Cols+col]
				}
			}
		}(col)
	}
	wg.Wait()
}
