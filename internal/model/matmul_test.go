package model

import (
	"fmt"
	"testing"
)

func TestRowParallelMatMul(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	y := make([]float32, 2*2)
	rowParallelMatMul(x, x, []int64{2, 3}, []int64{3, 2}, y)
	fmt.Println(y)
}

func TestColParallelMatMul(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	y := make([]float32, 2*2)
	colParallelMatMul(x, x, []int64{2, 3}, []int64{3, 2}, y)
	fmt.Println(y)
}
