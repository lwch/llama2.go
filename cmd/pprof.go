package cmd

import (
	"os"
	"runtime/pprof"
	"time"

	"github.com/lwch/runtime"
)

func profile() {
	f, err := os.Create("cpu.pprof")
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(pprof.StartCPUProfile(f))
	time.Sleep(time.Minute)
	pprof.StopCPUProfile()
	f.Close()
	os.Exit(0)
}

func memProfile() {
	f, err := os.Create("mem.pprof")
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(pprof.WriteHeapProfile(f))
	f.Close()
	// os.Exit(0)
	time.Sleep(time.Hour)
}
