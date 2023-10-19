package math

import (
	"fmt"
	"testing"
	"time"
)

func TestMatMul(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	y := make([]float32, 2*2)
	begin := time.Now()
	MatMul(x, x, 2, 2, 3, y)
	fmt.Println(y, time.Since(begin))
}
