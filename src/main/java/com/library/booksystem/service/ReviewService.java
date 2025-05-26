package com.library.booksystem.service;

import com.library.booksystem.dto.request.ReviewRequest;
import com.library.booksystem.dto.response.ReviewResponse;
import com.library.booksystem.exception.AppException;
import com.library.booksystem.exception.ErrorCode;
import com.library.booksystem.model.Book;
import com.library.booksystem.model.Review;
import com.library.booksystem.model.User;
import com.library.booksystem.repository.BookRepository;
import com.library.booksystem.repository.ReviewRepository;
import com.library.booksystem.repository.UserRepository;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class ReviewService {
    ReviewRepository reviewRepository;
    UserRepository userRepository;
    BookRepository bookRepository;

    @Transactional
    public ReviewResponse createReview(ReviewRequest request) {
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new AppException(ErrorCode.USER_NOT_EXIST));

        Book book = bookRepository.findById(request.getBookId())
                .orElseThrow(() -> new AppException(ErrorCode.BOOK_NOT_FOUND));

        Review review = new Review();
        review.setUser(user);
        review.setBook(book);
        review.setRating(request.getRating());
        review.setComment(request.getComment());
        review.setVisible(true);

        Review savedReview = reviewRepository.save(review);
        return toReviewResponse(savedReview);
    }

    public Page<ReviewResponse> getBookReviews(Integer bookId, Pageable pageable) {
        return reviewRepository.findByBookBookId(bookId, pageable)
                .map(this::toReviewResponse);
    }

    public Page<ReviewResponse> getUserReviews(String userId, Pageable pageable) {
        return reviewRepository.findByUserUserId(userId, pageable)
                .map(this::toReviewResponse);
    }

    @Transactional
    public void deleteReview(Integer reviewId) {
        Review review = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new AppException(ErrorCode.REVIEW_NOT_FOUND));
        reviewRepository.delete(review);
    }

    @Transactional
    public ReviewResponse toggleReviewVisibility(Integer reviewId) {
        Review review = reviewRepository.findById(reviewId)
                .orElseThrow(() -> new AppException(ErrorCode.REVIEW_NOT_FOUND));
        review.setVisible(!review.isVisible());
        return toReviewResponse(reviewRepository.save(review));
    }

    private ReviewResponse toReviewResponse(Review review) {
        return ReviewResponse.builder()
                .reviewId(review.getReviewId())
                .userId(review.getUser().getUserId())
                .username(review.getUser().getUsername())
                .bookId(review.getBook().getBookId())
                .bookTitle(review.getBook().getTitle())
                .rating(review.getRating())
                .comment(review.getComment())
                .createdAt(review.getCreatedAt())
                .isVisible(review.isVisible())
                .build();
    }
} 