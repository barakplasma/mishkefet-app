// Minimal static server for the offline image: serves /srv on :8080.
// net/http already handles Range requests and the .mjs/.wasm MIME types the app needs.
package main

import (
	"log"
	"net/http"
)

func main() {
	mux := http.NewServeMux()
	mux.Handle("/mishkefet-app/", http.FileServer(http.Dir("/srv")))
	// The app is built with base path /mishkefet-app/ upstream.
	mux.Handle("GET /{$}", http.RedirectHandler("/mishkefet-app/", http.StatusFound))
	log.Fatal(http.ListenAndServe(":8080", mux))
}
