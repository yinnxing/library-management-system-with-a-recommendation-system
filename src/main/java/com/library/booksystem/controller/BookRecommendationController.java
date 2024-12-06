package com.library.booksystem.controller;
import com.library.booksystem.service.BookRecommendationService;
import com.library.booksystem.service.BookService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class BookRecommendationController {

    BookRecommendationService recommendationService;
    BookService bookService;

    
@GetMapping("/recommend")
    public Mono<String> recommendBooks(@RequestParam("userId") String userId) {
        String lastBorrowedBook = bookService.getLastBorrowedBook(userId);
        return recommendationService.getRecommendedBooks(lastBorrowedBook);
    }
}
