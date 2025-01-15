package com.library.booksystem.controller;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.model.RecommendationFeedback;
import com.library.booksystem.service.BookRecommendationService;
import com.library.booksystem.service.BookService;
import com.library.booksystem.service.RecommendationFeedbackService;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
@RequiredArgsConstructor
@RequestMapping("/recommend")
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class BookRecommendationController {

    BookRecommendationService recommendationService;
    BookService bookService;
    RecommendationFeedbackService feedbackService;

    
@GetMapping
    public Mono<String> recommendBooks(@RequestParam("userId") String userId) {
        String lastBorrowedBook = bookService.getLastBorrowedBook(userId);
        return recommendationService.getRecommendedBooks(lastBorrowedBook);
    }
    @PostMapping("/feedback")
    public ApiResponse<Void> saveFeedback(@RequestBody RecommendationFeedback feedback){
        feedbackService.saveFeedback(feedback);
        return ApiResponse.<Void>builder()
                .message("save feedback successfully!")
                .build();

    }

}
