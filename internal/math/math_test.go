package math

import (
	"fmt"
	"testing"
)

func TestRowParallelMatMul(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	y := make([]float32, 2*2)
	rowParallelMatMul(x, x, 2, 2, 3, y)
	fmt.Println(y)
}

func TestColParallelMatMul(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	y := make([]float32, 2*2)
	colParallelMatMul(x, x, 2, 2, 3, y)
	fmt.Println(y)
}
