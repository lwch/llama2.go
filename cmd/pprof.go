package cmd

import (
	"os"
	"runtime/pprof"

	"github.com/lwch/runtime"
)

func cpuProfile() func() {
	f, err := os.Create("cpu.pprof")
	runtime.Assert(err)
	runtime.Assert(pprof.StartCPUProfile(f))
	return func() {
		pprof.StopCPUProfile()
		f.Close()
	}
}

func memProfile() {
	f, err := os.Create("mem.pprof")
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(pprof.WriteHeapProfile(f))
	// f.Close()
	// os.Exit(0)
	// time.Sleep(time.Hour)
}
