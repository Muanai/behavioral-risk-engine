FROM golang:1.26.4-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o feature_service main.go

FROM alpine:latest

WORKDIR /root/

COPY --from=builder /app/feature_service .

EXPOSE 8080

CMD ["./feature_service"]