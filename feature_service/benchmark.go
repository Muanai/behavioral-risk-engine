package main

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

func main() {
	totalRequests := 10000
	concurrency := 100
	url := "http://localhost:8080/features?user_id=2"

	fmt.Printf("Initializing benchmark: %d total requests at %d concurrency level...\n", totalRequests, concurrency)

	jobs := make(chan int, totalRequests)
	for i := 1; i <= totalRequests; i++ {
		jobs <- i
	}
	close(jobs)

	var wg sync.WaitGroup
	var mu sync.Mutex
	successCount := 0
	failCount := 0

	start := time.Now()

	for w := 1; w <= concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := &http.Client{
				Timeout: 5 * time.Second,
			}
			for range jobs {
				resp, err := client.Get(url)
				if err != nil {
					mu.Lock()
					failCount++
					mu.Unlock()
					continue
				}
				if resp.StatusCode == http.StatusOK {
					mu.Lock()
					successCount++
					mu.Unlock()
				} else {
					mu.Lock()
					failCount++
					mu.Unlock()
				}
				resp.Body.Close()
			}
		}()
	}

	wg.Wait()
	duration := time.Since(start)

	rps := float64(successCount) / duration.Seconds()
	avgLatency := time.Duration(duration.Nanoseconds() / int64(totalRequests))

	fmt.Println("\n--------------------------------------------------")
	fmt.Println("FEATURE SERVICE BENCHMARK REPORT")
	fmt.Printf("Total Requests   : %d\n", totalRequests)
	fmt.Printf("Concurrency      : %d\n", concurrency)
	fmt.Printf("Total Duration   : %v\n", duration)
	fmt.Printf("Success Requests : %d\n", successCount)
	fmt.Printf("Failed Requests  : %d\n", failCount)
	fmt.Printf("Throughput (RPS) : %.2f req/sec\n", rps)
	fmt.Printf("Avg Latency/Req  : %v\n", avgLatency)
	fmt.Println("--------------------------------------------------")
}