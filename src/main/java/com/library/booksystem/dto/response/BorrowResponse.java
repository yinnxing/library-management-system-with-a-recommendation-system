package com.library.booksystem.dto.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class BorrowResponse {
    String userId;
    Integer bookId;
    String transactionId;
    String bookTitle;
    LocalDateTime borrowDate;
    LocalDateTime pickupDeadline;
    String status;
    String message;
}
