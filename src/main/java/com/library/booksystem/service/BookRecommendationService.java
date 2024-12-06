package com.library.booksystem.service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.http.HttpStatus;
import reactor.core.publisher.Mono;

@Service
public class BookRecommendationService {


        private final WebClient webClient;

        public BookRecommendationService(WebClient.Builder webClientBuilder) {
            this.webClient = webClientBuilder.baseUrl("http://127.0.0.1:5000").build();
        }

        public Mono<String> getRecommendedBooks(String bookName) {
            return webClient.get()
                    .uri(uriBuilder -> uriBuilder.path("/recommend")
                            .queryParam("book_name", bookName)
                            .build())
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError(), response -> Mono.error(new Exception("Client error")))
                    .onStatus(status -> status.is5xxServerError(), response -> Mono.error(new Exception("Server error")))
                    .bodyToMono(String.class);
        }


}
