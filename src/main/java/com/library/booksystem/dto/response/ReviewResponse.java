package com.library.booksystem.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ReviewResponse {
    private Integer reviewId;
    private String userId;
    private String username;
    private Integer bookId;
    private String bookTitle;
    private Integer rating;
    private String comment;
    private LocalDateTime createdAt;
    private boolean isVisible;
} 