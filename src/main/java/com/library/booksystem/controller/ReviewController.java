package com.library.booksystem.controller;

import com.library.booksystem.dto.request.ReviewRequest;
import com.library.booksystem.dto.response.ApiResponse;
import com.library.booksystem.dto.response.ReviewResponse;
import com.library.booksystem.service.ReviewService;
import jakarta.validation.Valid;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/reviews")
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class ReviewController {
    ReviewService reviewService;

    @PostMapping
    public ApiResponse<ReviewResponse> createReview(@Valid @RequestBody ReviewRequest request) {
        return ApiResponse.<ReviewResponse>builder()
                .result(reviewService.createReview(request))
                .build();
    }

    @GetMapping("/book/{bookId}")
    public ApiResponse<Page<ReviewResponse>> getBookReviews(
            @PathVariable Integer bookId,
            Pageable pageable) {
        return ApiResponse.<Page<ReviewResponse>>builder()
                .result(reviewService.getBookReviews(bookId, pageable))
                .build();
    }

    @GetMapping("/user/{userId}")
    public ApiResponse<Page<ReviewResponse>> getUserReviews(
            @PathVariable String userId,
            Pageable pageable) {
        return ApiResponse.<Page<ReviewResponse>>builder()
                .result(reviewService.getUserReviews(userId, pageable))
                .build();
    }

    @DeleteMapping("/{reviewId}")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<Void> deleteReview(@PathVariable Integer reviewId) {
        reviewService.deleteReview(reviewId);
        return ApiResponse.<Void>builder().build();
    }

    @PatchMapping("/{reviewId}/visibility")
    @PreAuthorize("hasRole('ADMIN')")
    public ApiResponse<ReviewResponse> toggleReviewVisibility(@PathVariable Integer reviewId) {
        return ApiResponse.<ReviewResponse>builder()
                .result(reviewService.toggleReviewVisibility(reviewId))
                .build();
    }
} 