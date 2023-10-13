package checkpoint

import (
	"fmt"
	"testing"
	"unsafe"
)

func TestLoad(t *testing.T) {
	m, err := Load("consolidated.00.pth")
	if err != nil {
		t.Fatal(err)
	}
	// for name, p := range m.Params() {
	// 	fmt.Println(name, p.Shape())
	// }
	p := m.Params()["layers.0.feed_forward.w1.weight"]
	data, err := p.Load()
	if err != nil {
		t.Fatal(err)
	}
	dt := data.([]uint16)
	fmt.Println(
		decodeBFloat16(dt[0]),
		decodeBFloat16(dt[1]),
		decodeBFloat16(dt[2]),
		decodeBFloat16(dt[6912*5120-1]))
}

func decodeBFloat16(u16 uint16) float32 {
	n := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&n))
}
