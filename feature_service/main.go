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
	// Hanya naik satu tingkat ke root repositori
	err := godotenv.Load("../.env")
	if err != nil {
		log.Println("Peringatan: File .env tidak ditemukan, menggunakan fallback lokal.")
	}

	redisHost := os.Getenv("REDIS_HOST")
	// Paksa rute ke IPv4 jika sistem mengembalikan localhost kosong atau default
	if redisHost == "" || redisHost == "localhost" {
		redisHost = "127.0.0.1"
	}

	redisPort := os.Getenv("REDIS_PORT")
	if redisPort == "" {
		redisPort = "6379"
	}

	// Membangun jembatan absolut ke kontainer Redis
	rdb = redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", redisHost, redisPort),
	})
}

func getFeatureHandler(w http.ResponseWriter, r *http.Request) {
	// Menangkap parameter user_id dari URL
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id wajib diisi", http.StatusBadRequest)
		return
	}

	// Kunci pencarian diubah sesuai format pipeline Python sebelumnya
	redisKey := fmt.Sprintf("customer:%s", userID)

	// Merampas seluruh nilai hash fitur dari Redis
	features, err := rdb.HGetAll(ctx, redisKey).Result()
	if err != nil {
		http.Error(w, "Gagal menembus Redis", http.StatusInternalServerError)
		return
	}

	if len(features) == 0 {
		http.Error(w, "Fitur behavioral tidak ditemukan untuk nasabah ini", http.StatusNotFound)
		return
	}

	// Memuntahkan data dalam wujud JSON
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(features)
}

func main() {
	http.HandleFunc("/features", getFeatureHandler)

	port := ":8080"
	fmt.Printf("Feature Service API menyala dan mendengarkan di port %s...\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}
