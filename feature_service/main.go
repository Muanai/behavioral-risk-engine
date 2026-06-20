package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"
)

var ctx = context.Background()
var rdb *redis.Client

func init() {
	// Only go up one level to the repository root
	err := godotenv.Load("../.env")
	if err != nil {
		log.Println("Warning: .env file not found, using local fallback.")
	}

	redisHost := os.Getenv("REDIS_HOST")
	// Force IPv4 routing if system returns empty or default localhost
	if redisHost == "" || redisHost == "localhost" {
		redisHost = "127.0.0.1"
	}

	redisPort := os.Getenv("REDIS_PORT")
	if redisPort == "" {
		redisPort = "6379"
	}

	// Establish an absolute connection to the Redis container
	rdb = redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", redisHost, redisPort),
	})
}

func getFeatureHandler(w http.ResponseWriter, r *http.Request) {
	// Capture user_id parameter from URL
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	// Search key changed to match previous Python pipeline format
	redisKey := fmt.Sprintf("customer:%s", userID)

	// Retrieve all feature hash values from Redis
	features, err := rdb.HGetAll(ctx, redisKey).Result()
	if err != nil {
		http.Error(w, "Failed to connect to Redis", http.StatusInternalServerError)
		return
	}

	if len(features) == 0 {
		http.Error(w, "Behavioral features not found for this customer", http.StatusNotFound)
		return
	}

	// Output data in JSON format
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(features)
}

func main() {
	http.HandleFunc("/features", getFeatureHandler)

	port := ":8080"
	fmt.Printf("Feature Service API is running and listening on port %s...\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
