package checkpoint

import (
	"fmt"
	"testing"
)

func TestLoad(t *testing.T) {
	m, err := Load("consolidated.00.pth")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(m.Params())
}
